#include <vtkh/rendering/AutoCamera.hpp>
#include "vtkh/rendering/ScalarRenderer.hpp"
#include <vtkh/Error.hpp>

#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <viskores/VectorAnalysis.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/TryExecute.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/filter/density_estimate/worklet/FieldHistogram.h>
//take out
#include <viskores/io/VTKDataSetWriter.h>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

template <typename T>
void fibonacci_sphere(int i, int samples, T* points)
{
  int rnd = 1;
  //if randomize:
  //    rnd = random.random() * samples

  T offset = 2./samples;
  T increment = M_PI * (3. - sqrt(5.));

  T y = ((i * offset) - 1) + (offset / 2);
  T r = sqrt(1 - pow(y,2));

  T phi = ((i + rnd) % samples) * increment;

  T x = cos(phi) * r;
  T z = sin(phi) * r;

  points[0] = x;
  points[1] = y;
  points[2] = z;
}

void
GetCamera(int frame, int nframes, double diameter, float *lookat, double *cam_pos)
{
  double points[3];
  fibonacci_sphere<double>(frame, nframes, points);

  cam_pos[0] = (diameter*points[0]) + lookat[0];
  cam_pos[1] = (diameter*points[1]) + lookat[1];
  cam_pos[2] = (diameter*points[2]) + lookat[2];

  //std::cerr << "zoom: " << zoom << std::endl;
  //std::cerr << "diameter: " << diameter << std::endl;
  //std::cerr << "lookat: " << lookat[0] << " " << lookat[1] << " " << lookat[2] << std::endl;
  //std::cerr << "points: " << points[0] << " " << points[1] << " " << points[2] << std::endl;
  //std::cerr << "camera position: " << cam_pos[0] << " " << cam_pos[1] << " " << cam_pos[2] << std::endl;
}

struct print_f
{
  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<T,S> &a) const
  {
    viskores::Id s = a.GetNumberOfValues();
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
  std::vector<viskores::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<T> data;
     
  //if there is data: loop through domains and grab all triangles.
  if(!vtkhData.IsEmpty())
  {
    for(int i = 0; i < localDomainIds.size(); i++)
    {
      viskores::cont::DataSet dataset = vtkhData.GetDomainById(localDomainIds[i]);
      viskores::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      viskores::cont::UnknownCellSet cellset = dataset.GetCellSet();
      //Get variable
      viskores::cont::Field field = dataset.GetField(field_name);
      
      long int size = field.GetNumberOfValues();
      
      using data_d = viskores::cont::ArrayHandle<viskores::Float64>;
      using data_f = viskores::cont::ArrayHandle<viskores::Float32>;
      if(field.GetData().IsType<data_d>())
      {
        viskores::cont::ArrayHandle<viskores::Float64> field_data;
        field.GetData().AsArrayHandle(field_data);
        auto portal = field_data.ReadPortal();

        for(int i = 0; i < size; i++)
        {
          data.push_back(portal.Get(i));
        }
      }
      if(field.GetData().IsType<data_f>())
      {
        viskores::cont::ArrayHandle<viskores::Float64> field_data;
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
  inline VISKORES_EXEC_CONT T operator()(const T& numerator, const T& denominator) const
  {
    const T prob = numerator / denominator;
    if (prob == T(0))
    {
      return T(0);
    }
    return prob * viskores::Log(prob);
  }
};

template <typename T>
T calcEntropyMM(const viskores::cont::ArrayHandle<T>& data, int nBins, T min, T max)
{
  viskores::worklet::FieldHistogram worklet;
  viskores::cont::ArrayHandle<viskores::Id> hist;
  T stepSize;
  worklet.Run(data, nBins, min, max, stepSize, hist);

  auto len = viskores::cont::make_ArrayHandleConstant(
    static_cast<T>(data.GetNumberOfValues()), 
    hist.GetNumberOfValues());
  viskores::cont::ArrayHandle<T> subEntropies;
  viskores::cont::Algorithm::Transform(hist, len, subEntropies, CalculateEntropy<T>{});

  T entropy = viskores::cont::Algorithm::Reduce(subEntropies, T(0));

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
class CopyWithOffset : public viskores::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, WholeArrayInOut dest);
  using ExecutionSignature = void(InputIndex, _1, _2);

  VISKORES_CONT
  CopyWithOffset(const viskores::Id offset = 0)
      : Offset(offset)
  {
  }
  template <typename OutArrayType>
  VISKORES_EXEC inline void operator()(const viskores::Id idx, const FloatType &srcValue, OutArrayType &destArray) const
  {
    destArray.Set(idx + this->Offset, srcValue);
  }

private:
  viskores::Id Offset;
};

template <typename SrcType, typename DestType>
void copyArrayWithOffset(const viskores::cont::ArrayHandle<SrcType> &src, viskores::cont::ArrayHandle<DestType> &dest, viskores::Id offset)
{
  viskores::cont::Invoker invoker;
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

  VISKORES_EXEC_CONT inline T operator()(const T &a, const T &b) const
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

  VISKORES_EXEC_CONT inline bool IsValid(const T &t) const
  {
    return !viskores::IsNan(t) && t > MinValid && t < MaxValid;
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
struct CopyWithChecksMask : public viskores::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, FieldOut dest);
  using ExecutionSignature = void(_1, _2);

  VISKORES_CONT
  CopyWithChecksMask(DataCheckFlags checks, DataCheckVals<FloatType> checkVals)
      : Checks(checks),
        CheckVals(checkVals)
  {
  }

  VISKORES_EXEC inline void operator()(const FloatType &val, viskores::IdComponent& mask) const
  {
    bool passed = true;
    if(this->HasCheck(CheckNan))
    {
      passed = passed && !viskores::IsNan(val);   
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
  
  VISKORES_EXEC inline bool HasCheck(DataCheckFlags check) const
  {
    return (Checks & check) == check;
  }

  DataCheckFlags Checks;
  DataCheckVals<FloatType> CheckVals;
};

template<typename SrcType>
viskores::cont::ArrayHandle<SrcType> copyWithChecks(
  const viskores::cont::ArrayHandle<SrcType>& src, 
  DataCheckFlags checks, 
  DataCheckVals<SrcType> checkVals = DataCheckVals<SrcType>{})
{
  viskores::cont::ArrayHandle<viskores::IdComponent> mask;
  viskores::cont::Invoker invoker;
  invoker(CopyWithChecksMask<SrcType>(checks, checkVals), src, mask);
  
  viskores::cont::ArrayHandle<SrcType> dest;
  viskores::cont::Algorithm::CopyIf(src, mask, dest);
  return dest;
}

template <typename T>
viskores::cont::ArrayHandle<T>
GetScalarDataAsArrayHandle(vtkh::DataSet &vtkhData, std::string field_name)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<viskores::Id> localDomainIds = vtkhData.GetDomainIds();
  viskores::cont::ArrayHandle<T> totalFieldData;

  if (!vtkhData.IsEmpty())
  {
    // Loop once to get the total number of items and reserve the vector
    viskores::Id totalNumberOfValues = std::accumulate(
        localDomainIds.begin(),
        localDomainIds.end(),
        0,
        [&](const viskores::Id &acc, const viskores::Id domainId)
        {
          const viskores::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
          const viskores::cont::Field &field = dataset.GetField(field_name);
          return acc + field.GetData().GetNumberOfValues();
        });

    totalFieldData.Allocate(totalNumberOfValues);
    viskores::Id offset = 0;
    for (auto &domainId : localDomainIds)
    {
      const viskores::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
      const viskores::cont::Field &field = dataset.GetField(field_name);
      const auto fieldData = field.GetData().AsArrayHandle<viskores::cont::ArrayHandle<T>>();
      copyArrayWithOffset(fieldData, totalFieldData, offset);
      offset += fieldData.GetNumberOfValues();
    }
  }

  return totalFieldData;
}

double
calculateDataEntropy(vtkh::DataSet* dataset, std::string field_name, double field_min, double field_max, int bins)
{
  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif
//dataset->PrintSummary(std::cerr);
  using data_d = viskores::cont::ArrayHandle<viskores::Float64>;
  using data_f = viskores::cont::ArrayHandle<viskores::Float32>;
  
  if(rank == 0)
  {
    viskores::cont::Field field = dataset->GetField(field_name,0);

    if(field.GetData().IsType<data_d>())
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float64>(*dataset, field_name.c_str());
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckZero;
        field_data = copyWithChecks<viskores::Float64>(field_data, checks);
        entropy = calcEntropyMM<viskores::Float64>(field_data, bins, field_min, field_max);
      } 
      else
      {
        entropy = 0;
      }
    }
    else
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float32>(*dataset, field_name.c_str());
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckZero;
        field_data = copyWithChecks<viskores::Float32>(field_data, checks);
        entropy = calcEntropyMM<viskores::Float32>(field_data, bins, viskores::Float32(field_min), viskores::Float32(field_max));
      } 
      else
      {
        entropy = 0;
      }
    }
  }

  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double 
calculateDepthEntropy(vtkh::DataSet* dataset, std::string field_name, double diameter, int bins)
{

  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif

  using data_d = viskores::cont::ArrayHandle<viskores::Float64>;
  using data_f = viskores::cont::ArrayHandle<viskores::Float32>;

  if(rank == 0)
  {
    viskores::cont::Field field = dataset->GetField(field_name,0);

    if(field.GetData().IsType<data_d>())
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float64>(*dataset, "depth");
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
        DataCheckVals<viskores::Float64> checkVals; 
	checkVals.Min = 0;
       	checkVals.Max = viskores::Float64(INT_MAX);
        field_data = copyWithChecks<viskores::Float64>(field_data, checks, checkVals);
	viskores::Float64 min = 0.0;
        entropy = calcEntropyMM<viskores::Float64>(field_data, bins, min, diameter);
      } 
      else
      {
        entropy = 0;
      }
    }
    else
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float32>(*dataset, "depth");
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
        DataCheckVals<viskores::Float32> checkVals; 
	checkVals.Min = 0;
       	checkVals.Max = viskores::Float32(INT_MAX);
        field_data = copyWithChecks<viskores::Float32>(field_data, checks, checkVals);
	viskores::Float32 min = 0.0;
        entropy = calcEntropyMM<viskores::Float32>(field_data, bins, min, viskores::Float32(diameter));
      } 
      else
      {
        entropy = 0;
      }
    }
  }
  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double 
calculateShadingEntropy(vtkh::DataSet* dataset, std::string field_name, int bins)
{

  double entropy = 0.0;
  int rank = 0;
  #if VTKH_PARALLEL 
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);
  #endif

  using data_d = viskores::cont::ArrayHandle<viskores::Float64>;
  using data_f = viskores::cont::ArrayHandle<viskores::Float32>;

  if(rank == 0)
  {
    viskores::cont::Field field = dataset->GetField(field_name,0);

    if(field.GetData().IsType<data_d>())
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float64>(*dataset, "shading");
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
        DataCheckVals<viskores::Float64> checkVals; 
	checkVals.Min = 0;
       	checkVals.Max = viskores::Float64(INT_MAX);
        field_data = copyWithChecks<viskores::Float64>(field_data, checks, checkVals);
	viskores::Float32 min = 0.0;
	viskores::Float32 max = 1.0;
        entropy = calcEntropyMM<viskores::Float64>(field_data, bins, min, max);
      } 
      else
      {
        entropy = 0;
      }
    }
    else
    {
      auto field_data = GetScalarDataAsArrayHandle<viskores::Float32>(*dataset, "shading");
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
        DataCheckVals<viskores::Float32> checkVals; 
	checkVals.Min = 0;
       	checkVals.Max = viskores::Float32(INT_MAX);
        field_data = copyWithChecks<viskores::Float32>(field_data, checks, checkVals);
	viskores::Float32 min = 0.0;
	viskores::Float32 max = 1.0;
        entropy = calcEntropyMM<viskores::Float32>(field_data, bins, min, max);
      } 
      else
      {
        entropy = 0;
      }
    }
  }
  #if VTKH_PARALLEL
  MPI_Bcast(&entropy, 1, MPI_DOUBLE, 0, mpi_comm);
  #endif
  return entropy;
}

double
calculateMetricScore(vtkh::DataSet* dataset, std::string metric, std::string field_name, viskores::Float64 field_min, viskores::Float64 field_max, double diameter, int bins)
{
  double score = 0.0;

  if(metric == "data_entropy")
  {
    score = calculateDataEntropy(dataset, field_name, field_min, field_max, bins);
  }
  else if (metric == "dds_entropy")
  {
    double shading_score = calculateShadingEntropy(dataset, field_name, bins);
    double data_score = calculateDataEntropy(dataset, field_name, field_min, field_max, bins);
    double depth_score = calculateDepthEntropy(dataset, field_name, diameter, bins);
    score = shading_score+data_score+depth_score;
  }
  else if (metric == "shading_entropy")
  {
    score = calculateShadingEntropy(dataset, field_name, bins);
  }
  else if (metric == "depth_entropy")
  {
    score = calculateDepthEntropy(dataset, field_name, diameter, bins);
  }
  else
  {
    std::stringstream msg;
    msg<< "This metric '" << metric << "' is not supported. \n";
    throw Error(msg.str());
  }
  return score;
}

void
calculateDiameter(viskores::Bounds bounds, double &diameter)
{
  viskores::Float64 xb = viskores::Float64(bounds.X.Length());
  viskores::Float64 yb = viskores::Float64(bounds.Y.Length());
  viskores::Float64 zb = viskores::Float64(bounds.Z.Length());
  diameter = sqrt(xb*xb + yb*yb + zb*zb);

}

} // namespace detail

AutoCamera::AutoCamera()
  : m_bins(256),
    m_height(1024),
    m_width(1024)
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
AutoCamera::SetNumBins(int bins)
{
  m_bins = bins;
}

int
AutoCamera::GetNumBins()
{
  return m_bins;
}

void 
AutoCamera::SetHeight(int height)
{
  m_height = height;
}

int
AutoCamera::GetHeight()
{
  return m_height;
}

void 
AutoCamera::SetWidth(int width)
{
  m_width = width;
}

int
AutoCamera::GetWidth()
{
  return m_width;
}

viskoresCamera
AutoCamera::GetCamera()
{
  return m_camera;
}

void
AutoCamera::PreExecute()
{
  Filter::PreExecute();
}

void
AutoCamera::DoExecute()
{

  int rank = 0;
  int world_size = 0;
  #if VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_size(mpi_comm, &world_size);
  MPI_Comm_rank(mpi_comm, &rank);
  #endif
  
  viskores::Range range = this->m_input->GetGlobalRange(m_field).ReadPortal().Get(0);
  viskores::Float64 field_min = range.Min;
  viskores::Float64 field_max = range.Max;

  viskores::Bounds g_bounds = this->m_input->GetGlobalBounds();
  double diameter = 0.0;
  detail::calculateDiameter(g_bounds, diameter);

  viskoresCamera *camera = new viskoresCamera;
  camera->ResetToBounds(g_bounds);
  viskores::Vec<viskores::Float32,3> lookat = camera->GetLookAt();
  float focus[3] = {lookat[0],lookat[1],lookat[2]};

  double winning_score  = -1;
  double losing_score   = 10000;
  int   winning_sample = -1;
  int   losing_sample  = -1;

  int count = 0;

  //loop through number of camera samples.
  for(int sample = 0; sample < m_samples; sample++)
  {
  /*================ Scalar Renderer Code ======================*/

    double cam_pos[3];
    detail::GetCamera(sample, m_samples, diameter, focus, cam_pos);
    viskores::Vec<viskores::Float64, 3> pos{cam_pos[0],
                            cam_pos[1],
                            cam_pos[2]};

    camera->SetPosition(pos);
    vtkh::ScalarRenderer tracer;
    tracer.SetWidth(m_width);
    tracer.SetHeight(m_height);
    tracer.SetInput(this->m_input); //vtkh dataset by toponame
    tracer.SetCamera(*camera);
    tracer.Update();

    vtkh::DataSet *output = tracer.GetOutput();
    //output->PrintSummary(std::cerr);

    double score = detail::calculateMetricScore(output, m_metric, m_field, 
						field_min, field_max, diameter, 
						m_bins);
    
    //std::cerr << "sample " << sample << " score: " << score << std::endl;

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

  //std::cerr << "winner is sample " << winning_sample << " with score: " << winning_score << std::endl;

  double best_c[3];
  detail::GetCamera(winning_sample, m_samples, diameter, focus, best_c);

  viskores::Vec<viskores::Float64, 3> pos{best_c[0], 
				best_c[1], 
				best_c[2]}; 
  camera->SetPosition(pos);
  m_camera = *camera;

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
