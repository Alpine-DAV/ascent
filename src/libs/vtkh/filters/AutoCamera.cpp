#include <vtkh/filters/AutoCamera.hpp>
#include <vtkh/rendering/ScalarRenderer.hpp>
#include <vtkh/Error.hpp>

#include <flow_graph.hpp>

#include <float.h>

#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/TryExecute.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/filter/density_estimate/worklet/FieldHistogram.h>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

void fibonacci_sphere(int i, int samples, double* points)
{
  int rnd = 1;
  //if randomize:
  //    rnd = random.random() * samples

  double offset = 2./samples;
  double increment = M_PI * (3. - sqrt(5.));

  double y = ((i * offset) - 1) + (offset / 2);
  double r = sqrt(1 - pow(y,2));

  double phi = ((i + rnd) % samples) * increment;

  double x = cos(phi) * r;
  double z = sin(phi) * r;

  points[0] = x;
  points[1] = y;
  points[2] = z;
}

void
GetCamera(int frame, int nframes, double radius, double* lookat, double *bounds, double *cam_pos)
{
  double points[3];
  fibonacci_sphere(frame, nframes, points);
  double zoom = 3.0;
  double near = zoom/8;
  double far = zoom*5;
  double angle = M_PI/6;

/*  if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  {
    if(points[2] >= 0)
      points[2] += radius;
    if(points[2] < 0)
      points[2] -= radius;
  }*/
  /*
  double x = (bounds[0] + bounds[1])/2;
  double y = (bounds[2] + bounds[3])/2;
  double z = (bounds[4] + bounds[5])/2;
  */ 

  cam_pos[0] = (zoom*radius*points[0]) + lookat[0];
  cam_pos[1] = (zoom*radius*points[1]) + lookat[1];
  cam_pos[2] = (zoom*radius*points[2]) + lookat[2];

  //cerr << "radius: " << radius << endl;
  //cerr << "lookat: " << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
  //cerr << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
}

struct print_f
{
  template<typename T, typename S>
  void operator()(const vtkm::cont::ArrayHandle<T,S> &a) const
  {
    vtkm::Id s = a.GetNumberOfValues();
    auto p = a.ReadPortal();
    for(int i = 0; i < s; ++i)
    {
      std::cout<<p.Get(i)<<" ";
    }
    std::cout<<"\n";
  }
};


template <typename T>
std::vector<T>
GetScalarData(vtkh::DataSet &vtkhData, const char *field_name)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<T> data;
     
  //if there is data: loop through domains and grab all triangles.
  if(!vtkhData.IsEmpty())
  {
    for(int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet dataset = vtkhData.GetDomainById(localDomainIds[i]);
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      vtkm::cont::UnknownCellSet cellset = dataset.GetCellSet();
      //Get variable
      vtkm::cont::Field field = dataset.GetField(field_name);
      
      long int size = field.GetNumberOfValues();
      
      using data_d = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using data_f = vtkm::cont::ArrayHandle<vtkm::Float64>;
      if(field.GetData().IsType<data_d>())
      {
        vtkm::cont::ArrayHandle<vtkm::Float64> field_data;
        field.GetData().AsArrayHandle(field_data);
        auto portal = field_data.ReadPortal();

        for(int i = 0; i < size; i++)
        {
          data.push_back(portal.Get(i));
        }
      }
      if(field.GetData().IsType<data_f>())
      {
        vtkm::cont::ArrayHandle<vtkm::Float64> field_data;
        field.GetData().AsArrayHandle(field_data);
        auto portal = field_data.ReadPortal();

        for(int i = 0; i < size; i++)
        {
          data.push_back(portal.Get(i));
        }
      }
    }
  }
  //else
    //cerr << "VTKH Data is empty" << endl;
  return data;
}

template <typename T>
struct CalculateEntropy
{
  inline VTKM_EXEC_CONT T operator()(const T& numerator, const T& denominator) const
  {
    const T prob = numerator / denominator;
    if (prob == T(0))
    {
      return T(0);
    }
    return prob * vtkm::Log(prob);
  }
};

template <typename T>
T calcEntropyMM(const vtkm::cont::ArrayHandle<T>& data, int nBins, T max, T min)
{
  vtkm::worklet::FieldHistogram worklet;
  vtkm::cont::ArrayHandle<vtkm::Id> hist;
  T stepSize;
  worklet.Run(data, nBins, min, max, stepSize, hist);

  auto len = vtkm::cont::make_ArrayHandleConstant(
    static_cast<T>(data.GetNumberOfValues()), 
    hist.GetNumberOfValues());
  vtkm::cont::ArrayHandle<T> subEntropies;
  vtkm::cont::Algorithm::Transform(hist, len, subEntropies, CalculateEntropy<T>{});

  T entropy = vtkm::cont::Algorithm::Reduce(subEntropies, T(0));

  return (entropy * -1.0);
}

template< typename T >
T calcEntropyMM( const std::vector<T> array, long len, int nBins , T field_min, T field_max)
{
  T min = field_min;
  T max = field_max;

  T stepSize = (max-min) / (T)nBins;
  if(stepSize == 0)
    return 0.0;

  long* hist = new long[ nBins ];
  for(int i = 0; i < nBins; i++ )
    hist[i] = 0;

  for(long i = 0; i < len; i++ )
  {
    T idx = (std::abs(array[i]) - min) / stepSize;
    if((int)idx == nBins )
      idx -= 1.0;
    hist[(int)idx]++;
  }

  T entropy = 0.0;
  for(int i = 0; i < nBins; i++ )
  {
    T prob = (T)hist[i] / (T)len;
    if(prob != 0.0 )
      entropy += prob * std::log( prob );
  }

  delete[] hist;

  return (entropy * -1.0);
}

template <typename FloatType>
class CopyWithOffset : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, WholeArrayInOut dest);
  using ExecutionSignature = void(InputIndex, _1, _2);

  VTKM_CONT
  CopyWithOffset(const vtkm::Id offset = 0)
      : Offset(offset)
  {
  }
  template <typename OutArrayType>
  VTKM_EXEC inline void operator()(const vtkm::Id idx, const FloatType &srcValue, OutArrayType &destArray) const
  {
    destArray.Set(idx + this->Offset, srcValue);
  }

private:
  vtkm::Id Offset;
};

template <typename SrcType, typename DestType>
void copyArrayWithOffset(const vtkm::cont::ArrayHandle<SrcType> &src, vtkm::cont::ArrayHandle<DestType> &dest, vtkm::Id offset)
{
  vtkm::cont::Invoker invoker;
  invoker(CopyWithOffset<SrcType>(offset), src, dest);
}

template <typename T>
struct MaxValueWithChecks
{
  MaxValueWithChecks(T minValid, T maxValid)
      : MinValid(minValid),
        MaxValid(maxValid)
  {
  }

  VTKM_EXEC_CONT inline T operator()(const T &a, const T &b) const
  {
    if (this->IsValid(a) && this->IsValid(b))
    {
      return (a > b) ? a : b;
    }
    else if (!this->IsValid(a))
    {
      return b;
    }
    else if (!this->IsValid(b))
    {
      return a;
    }
    else
    {
      return this->MinValid;
    }
  }

  VTKM_EXEC_CONT inline bool IsValid(const T &t) const
  {
    return !vtkm::IsNan(t) && t > MinValid && t < MaxValid;
  }

  T MinValid;
  T MaxValid;
};


enum DataCheckFlags
{
  CheckNan          = 1 << 0,
  CheckZero         = 1 << 1,
  CheckMinExclusive = 1 << 2,
  CheckMaxExclusive = 1 << 3,
};

template<typename T>
struct DataCheckVals
{
  T Min;
  T Max;
};

inline DataCheckFlags operator|(DataCheckFlags lhs, DataCheckFlags rhs)
{
  return static_cast<DataCheckFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

template <typename FloatType>
struct CopyWithChecksMask : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, FieldOut dest);
  using ExecutionSignature = void(_1, _2);

  VTKM_CONT
  CopyWithChecksMask(DataCheckFlags checks, DataCheckVals<FloatType> checkVals)
      : Checks(checks),
        CheckVals(checkVals)
  {
  }

  VTKM_EXEC inline void operator()(const FloatType &val, vtkm::IdComponent& mask) const
  {
    bool passed = true;
    if(this->HasCheck(CheckNan))
    {
      passed = passed && !vtkm::IsNan(val);   
    }
    if(this->HasCheck(CheckZero)) 
    {
      passed = passed && (val != FloatType(0));
    }
    if(this->HasCheck(CheckMinExclusive))
    {
      passed = passed && (val > this->CheckVals.Min);
    }
    if(this->HasCheck(CheckMaxExclusive))
    {
      passed = passed && (val < this->CheckVals.Max);
    }

    mask = passed ? 1 : 0;
  }
  
  VTKM_EXEC inline bool HasCheck(DataCheckFlags check) const
  {
    return (Checks & check) == check;
  }

  DataCheckFlags Checks;
  DataCheckVals<FloatType> CheckVals;
};

template<typename SrcType>
vtkm::cont::ArrayHandle<SrcType> copyWithChecks(
  const vtkm::cont::ArrayHandle<SrcType>& src, 
  DataCheckFlags checks, 
  DataCheckVals<SrcType> checkVals = DataCheckVals<SrcType>{})
{
  vtkm::cont::ArrayHandle<vtkm::IdComponent> mask;
  vtkm::cont::Invoker invoker;
  invoker(CopyWithChecksMask<SrcType>(checks, checkVals), src, mask);
  
  vtkm::cont::ArrayHandle<SrcType> dest;
  vtkm::cont::Algorithm::CopyIf(src, mask, dest);
  return dest;
}

template <typename T>
vtkm::cont::ArrayHandle<T>
GetScalarDataAsArrayHandle(vtkh::DataSet &vtkhData, std::string field_name)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  vtkm::cont::ArrayHandle<T> totalFieldData;

  if (!vtkhData.IsEmpty())
  {
    // Loop once to get the total number of items and reserve the vector
    vtkm::Id totalNumberOfValues = std::accumulate(
        localDomainIds.begin(),
        localDomainIds.end(),
        0,
        [&](const vtkm::Id &acc, const vtkm::Id domainId)
        {
          const vtkm::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
          const vtkm::cont::Field &field = dataset.GetField(field_name);

          return acc + field.GetData().GetNumberOfValues();
        });

    totalFieldData.Allocate(totalNumberOfValues);
    vtkm::Id offset = 0;
    for (auto &domainId : localDomainIds)
    {
      const vtkm::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
      const vtkm::cont::Field &field = dataset.GetField(field_name);
      const auto fieldData = field.GetData().AsArrayHandle<vtkm::cont::ArrayHandle<T>>();
      copyArrayWithOffset(fieldData, totalFieldData, offset);
      offset += fieldData.GetNumberOfValues();
    }
  }

  return totalFieldData;
}

double
calculateDataEntropy(vtkh::DataSet* dataset, int height, int width,std::string field_name, double field_min, double field_max)
{
  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif

  if(rank == 0)
  {
    auto field_data = GetScalarDataAsArrayHandle<double>(*dataset, field_name.c_str());
    if (field_data.GetNumberOfValues() > 0) 
    {
      DataCheckFlags checks = CheckNan | CheckZero;
      field_data = copyWithChecks<double>(field_data, checks);
      entropy = calcEntropyMM(field_data, 1000, field_min, field_max);
    } 
    else
    {
      entropy = 0;
    }
  }

  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double 
calculateDepthEntropy(vtkh::DataSet* dataset, int height, int width, double diameter)
{

  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif

  if(rank == 0)
  {
    auto field_data = GetScalarDataAsArrayHandle<double>(*dataset, "depth");
    if (field_data.GetNumberOfValues() > 0) 
    {
      DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
      DataCheckVals<double> checkVals { .Min = 0, .Max = double(INT_MAX) };
      field_data = copyWithChecks<double>(field_data, checks, checkVals);
      entropy = calcEntropyMM(field_data, 1000, double(0.0), diameter);
    } 
    else
    {
      entropy = 0;
    }
  }
  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double 
calculateShadingEntropy(vtkh::DataSet* dataset, int height, int width)
{

  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif

  if(rank == 0)
  {
    auto field_data = GetScalarDataAsArrayHandle<double>(*dataset, "shading");
    if (field_data.GetNumberOfValues() > 0) 
    {
      DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
      DataCheckVals<double> checkVals { .Min = 0, .Max = double(INT_MAX) };
      field_data = copyWithChecks<double>(field_data, checks, checkVals);
      entropy = calcEntropyMM(field_data, 1000, double(0.0), double(1.0));
    } 
    else
    {
      entropy = 0;
    }
  }
  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double
calculateMetricScore(vtkh::DataSet* dataset, std::string metric, std::string field_name, int height, int width, double field_min, double field_max, double diameter)
{
  double score = 0.0;

  if(metric == "data_entropy")
  {
    score = calculateDataEntropy(dataset, height, width, field_name, field_min, field_max);
  }
  else if (metric == "dds_entropy")
  {
    double shading_score = calculateShadingEntropy(dataset, height, width);
    double data_score = calculateDataEntropy(dataset, height, width, field_name, field_min, field_max);
    double depth_score = calculateDepthEntropy(dataset, height, width, diameter);
    score = shading_score+data_score+depth_score;
  }
  else if (metric == "shading_entropy")
  {
    score = calculateShadingEntropy(dataset, height, width);
  }
  else if (metric == "depth_entropy")
  {
    score = calculateDepthEntropy(dataset, height, width, diameter);
  }
  else
  {
    std::stringstream msg;
    msg<< "This metric is not supported. \n";
    throw Error(msg.str());
  }
  return score;
}

void
calculateDiameter(vtkm::Bounds bounds, double &diameter, double &radius)
{
  vtkm::Float64 xb = vtkm::Float64(bounds.X.Length());
  vtkm::Float64 yb = vtkm::Float64(bounds.Y.Length());
  vtkm::Float64 zb = vtkm::Float64(bounds.Z.Length());
  radius = sqrt(xb*xb + yb*yb + zb*zb)/2.0;
  diameter = sqrt(xb*xb + yb*yb + zb*zb)*6.0;

}

} // namespace detail

AutoCamera::AutoCamera()
{

}

AutoCamera::~AutoCamera()
{

}

void
AutoCamera::SetMetric(std::string metric)
{
  m_metric = metric;
}

std::string
AutoCamera::GetMetric()
{
  return m_metric;
}

void
AutoCamera::SetField(std::string field)
{
  m_field = field;
}

std::string
AutoCamera::GetField()
{
  return m_field;
}

void 
AutoCamera::SetNumSamples(int samples)
{
  m_samples = samples;
}

int
AutoCamera::GetNumSamples()
{
  return m_samples;
}

void
AutoCamera::PreExecute()
{
  Filter::PreExecute();
}

void
AutoCamera::DoExecute()
{
  int width = 1000;
  int height = 1000;

  std::vector<double> field_data;
  field_data = detail::GetScalarData<double>(*this->m_input, m_field.c_str());

  double field_min = 0.;
  double field_max = 0.;
  int rank = 0;
  int world_size = 0;
  #if VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_size(mpi_comm, &world_size);
  MPI_Comm_rank(mpi_comm, &rank);
  double local_field_min = 0.;
  double local_field_max = 0.;
  
  if(field_data.size())
  {
    local_field_min = (double)*min_element(field_data.begin(),field_data.end());
    local_field_max = (double)*max_element(field_data.begin(),field_data.end());
  }
  MPI_Reduce(&local_field_min, &field_min, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
  MPI_Reduce(&local_field_max, &field_max, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
  #else
  if(field_data.size())
  {
    field_min = (double)*min_element(field_data.begin(),field_data.end());
    field_max = (double)*max_element(field_data.begin(),field_data.end());
  }
  #endif


  vtkm::Bounds g_bounds = this->m_input->GetGlobalBounds();
  double radius = 0;
  double diameter = 0;
  detail::calculateDiameter(g_bounds, radius, diameter);
  double bounds[6] = {(double)g_bounds.X.Max, (double)g_bounds.X.Min, 
			(double)g_bounds.Y.Max, (double)g_bounds.Y.Min, 
	                (double)g_bounds.Z.Max, (double)g_bounds.Z.Min};

  vtkmCamera *camera = new vtkmCamera;
  camera->ResetToBounds(g_bounds);
  vtkm::Vec<vtkm::Float64,3> lookat = camera->GetLookAt();
  double focus[3] = {(double)lookat[0],(double)lookat[1],(double)lookat[2]};

  double winning_score  = -DBL_MAX;
  double losing_score   = DBL_MAX;
  int   winning_sample = -1;
  int   losing_sample  = -1;

  int count = 0;

  //loop through number of camera samples.
  for(int sample = 0; sample < m_samples; sample++)
  {
  /*================ Scalar Renderer Code ======================*/

    double cam_pos[3];
    detail::GetCamera(sample, m_samples, radius, focus, bounds, cam_pos);
    vtkm::Vec<vtkm::Float64, 3> pos{cam_pos[0],
                            cam_pos[1],
                            cam_pos[2]};

    camera->SetPosition(pos);
    vtkh::ScalarRenderer tracer;
    tracer.SetWidth(width);
    tracer.SetHeight(height);
    tracer.SetInput(this->m_input); //vtkh dataset by toponame
    tracer.SetCamera(*camera);
    tracer.Update();

    vtkh::DataSet *output = tracer.GetOutput();
    //output->PrintSummary(std::cerr);

    double score = detail::calculateMetricScore(output, m_metric, m_field, 
						height, width, field_min, 
						field_max, diameter);
    

    delete output;

  /*================ End Scalar Renderer  ======================*/

    //original
    if(winning_score < score)
    {
      winning_score = score;
      winning_sample = sample;
    }
    if(losing_score > score)
    {
      losing_score = score;
      losing_sample = sample;
    }
    count++;
  } //end of sample loop

  if(winning_sample == -1)
  {
    std::stringstream msg;
    msg<<"Something went terribly wrong; No camera position was chosen\n";
    throw Error(msg.str());
  }

  double best_c[3];
  detail::GetCamera(winning_sample, m_samples, radius, focus, bounds, best_c);

  vtkm::Vec<vtkm::Float64, 3> pos{best_c[0], 
				best_c[1], 
				best_c[2]}; 
  camera->SetPosition(pos);

  if(!graph().workspace().registry().has_entry("camera"))
  {
    graph().workspace().registry().add<vtkm::rendering::Camera>("camera",camera,1);
  }

  this->m_output = this->m_input;
}

void
AutoCamera::PostExecute()
{
  Filter::PostExecute();
}

std::string
AutoCamera::GetName() const
{
  return "vtkh::AutoCamera";
}

} // namespace vtkh