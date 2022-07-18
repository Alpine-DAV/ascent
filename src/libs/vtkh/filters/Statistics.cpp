#include <vtkh/filters/Statistics.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vector>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

class CopyToFloat : public vtkm::worklet::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2);

  VTKM_CONT CopyToFloat()
  { }

  template <typename T>
  VTKM_EXEC void operator()(const T &in, vtkm::Float32 &out) const
  {
    out = static_cast<vtkm::Float32>(in);
  }
};

class SubtractMean : public vtkm::worklet::WorkletMapField
{
public:
  // typedef void ControlSignature()
  using ControlSignature = void(FieldIn, FieldOut);
  using ExecutionSignature = _2(_1);
  using InputDomain = _1;

  vtkm::Float32 mean;

  // define constructor to set mean as the input mean
  VTKM_CONT SubtractMean(vtkm::Float32 mean) : mean(mean) { }

  template <typename T>
  VTKM_EXEC vtkm::Float32 operator()(T value) const
  {
    return static_cast<vtkm::Float32>(value - this->mean);
  }
};
// create worklet class to exponentiate vector elements by power
// Create a worklet
class VecPow : public vtkm::worklet::WorkletMapField
{
public:
  // take each element of the vector to the power p
  // typedef void ControlSignature()
  using ControlSignature = void(FieldIn, FieldOut);
  using ExecutionSignature = _2(_1);
  using InputDomain = _1;

  int power;

  // define constructor to set mean as the input mean
  VTKM_CONT VecPow(int p) : power(p) { }

  template <typename T>
  VTKM_EXEC vtkm::Float32 operator()(T value) const
  {
    return static_cast<vtkm::Float32>(vtkm::Pow(value,this->power));
  }
};

// worklet class to divide by scalar
class DivideByScalar : public vtkm::worklet::WorkletMapField
{
public:
  // typedef void ControlSignature()
  using ControlSignature = void(FieldIn, FieldOut);
  using ExecutionSignature = _2(_1);
  using InputDomain = _1;

  vtkm::Float32 scale;

  // define constructor to set scale as the input scale
  VTKM_CONT DivideByScalar(vtkm::Float32 s) : scale(s) { }

  template <typename T>
  VTKM_EXEC vtkm::Float32 operator()(T value) const
  {
    return static_cast<vtkm::Float32>(value/this->scale);
  }
};

} // namespace detail

Statistics::Statistics()
{

}

Statistics::~Statistics()
{

}

Statistics::Result Statistics::Run(vtkh::DataSet &data_set, const std::string field_name)
{
  VTKH_DATA_OPEN("statistics");
  VTKH_DATA_ADD("device", GetCurrentDevice());
  VTKH_DATA_ADD("input_cells", data_set.GetNumberOfCells());
  VTKH_DATA_ADD("input_domains", data_set.GetNumberOfDomains());
  const int num_domains = data_set.GetNumberOfDomains();

  if(!data_set.GlobalFieldExists(field_name))
  {
    throw Error("Statistics: field : '"+field_name+"' does not exist'");
  }
  std::vector<vtkm::cont::ArrayHandle<vtkm::Float32>> fields;
  int total_values = 0;

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    data_set.GetDomain(i, dom, domain_id);
    if(dom.HasField(field_name))
    {
      vtkm::cont::Field field = dom.GetField(field_name);
      vtkm::cont::ArrayHandle<vtkm::Float32> float_field;
      vtkm::worklet::DispatcherMapField<detail::CopyToFloat>(detail::CopyToFloat())
        .Invoke(field.GetData().ResetTypes(vtkm::TypeListFieldScalar(),VTKM_DEFAULT_STORAGE_LIST{}), float_field);
      fields.push_back(float_field);
      total_values += float_field.GetNumberOfValues();
    }
  }

  Statistics::Result res;
#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce(MPI_IN_PLACE, &total_values, 1, MPI_INT, MPI_SUM, mpi_comm);
#endif

  // mean
  vtkm::Float32 sum = 0;
  for(size_t i = 0; i < fields.size(); ++i)
  {
    sum += vtkm::cont::Algorithm::Reduce(fields[i],0.0f,vtkm::Sum());
  }

#ifdef VTKH_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
#endif
  vtkm::Float32 mean = sum / vtkm::Float32(total_values);
  res.mean = mean;

  std::vector<vtkm::cont::ArrayHandle<vtkm::Float32>> unbiased_fields(fields.size());
  std::vector<vtkm::cont::ArrayHandle<vtkm::Float32>> temp_fields(fields.size());

  vtkm::cont::Invoker invoke;

  // variance
  vtkm::Float32 sum2 = 0;
  for(size_t i = 0; i < fields.size(); ++i)
  {
    invoke(detail::SubtractMean(mean), fields[i], unbiased_fields[i]);
    invoke(detail::VecPow(2),unbiased_fields[i],temp_fields[i]);
    sum2 += vtkm::cont::Algorithm::Reduce(temp_fields[i],0.0f,vtkm::Sum());
  }

#ifdef VTKH_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum2, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
#endif

  vtkm::Float32 variance = sum2 / vtkm::Float32(total_values - 1);
  res.variance = variance;

  // skewness
  vtkm::Float32 sum3 = 0;
  for(size_t i = 0; i < fields.size(); ++i)
  {
    invoke(detail::VecPow(3),unbiased_fields[i],temp_fields[i]);
    sum3 += vtkm::cont::Algorithm::Reduce(temp_fields[i],0.0f,vtkm::Sum());
  }

#ifdef VTKH_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum3, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
#endif
  vtkm::Float32 skewness = (sum3/vtkm::Float32(total_values))/(vtkm::Pow(variance,1.5f));
  res.skewness = skewness;

  // kertosis
  vtkm::Float32 sum4 = 0;
  for(size_t i = 0; i < fields.size(); ++i)
  {
    invoke(detail::VecPow(4),unbiased_fields[i],temp_fields[i]);
    sum4 += vtkm::cont::Algorithm::Reduce(temp_fields[i],0.0f,vtkm::Sum());
  }

#ifdef VTKH_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum4, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
#endif
  vtkm::Float32 kurtosis = (sum4/vtkm::Float32(total_values))/vtkm::Pow(variance,2.f) - 3.0f;
  res.kurtosis = kurtosis;

  VTKH_DATA_CLOSE();
  return res;
}


} //  namespace vtkh
