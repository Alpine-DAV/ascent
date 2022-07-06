#include "Log.hpp"
#include <vtkh/Error.hpp>

#include <vtkm/Math.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkh
{

namespace detail
{
class LogField : public vtkm::worklet::WorkletMapField
{
  const vtkm::Float32 m_min_value;
public:
  VTKM_CONT
  LogField(const vtkm::Float32 min_value)
   : m_min_value(min_value)
  {}

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const T &value, vtkm::Float32& log_value) const
  {
    vtkm::Float32 f_value = static_cast<vtkm::Float32>(value);
    f_value = vtkm::Max(m_min_value, f_value);
    log_value = vtkm::Log(f_value);
  }
}; //class LogField

class Log10Field : public vtkm::worklet::WorkletMapField
{
  const vtkm::Float32 m_min_value;
public:
  VTKM_CONT
  Log10Field(const vtkm::Float32 min_value)
   : m_min_value(min_value)
  {}

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const T &value, vtkm::Float32& log_value) const
  {
    vtkm::Float32 f_value = static_cast<vtkm::Float32>(value);
    f_value = vtkm::Max(m_min_value, f_value);
    log_value = vtkm::Log10(f_value);
  }
}; //class Log10Field

class Log2Field : public vtkm::worklet::WorkletMapField
{
  const vtkm::Float32 m_min_value;
public:
  VTKM_CONT
  Log2Field(const vtkm::Float32 min_value)
   : m_min_value(min_value)
  {}

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const T &value, vtkm::Float32& log_value) const
  {
    vtkm::Float32 f_value = static_cast<vtkm::Float32>(value);
    f_value = vtkm::Max(m_min_value, f_value);
    log_value = vtkm::Log2(f_value);
  }
}; //class Log2Field

} // namespace detail

Log::Log()
  : m_min_value(0.0001f),
    m_clamp_to_min(false)
{
}

Log::~Log()
{

}

void
Log::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Log::SetClampToMin(bool on)
{
  m_clamp_to_min = on;
}

void
Log::SetClampMin(vtkm::Float32 min_value)
{

  if(min_value <= 0)
  {
    throw Error("Log: min clamp value must be positive");
  }

  m_min_value = min_value;
}

void
Log::SetResultField(const std::string &field_name)
{
  m_result_name = field_name;
}


std::string
Log::GetField() const
{
  return m_field_name;
}

std::string
Log::GetResultField() const
{
  return m_result_name;
}

void Log::PreExecute()
{
  Filter::PreExecute();

  Filter::CheckForRequiredField(m_field_name);

  if(m_result_name== "")
  {
    m_result_name= "log(" + m_field_name + ")";
  }

}

void Log::PostExecute()
{
  Filter::PostExecute();
}

void Log::DoExecute()
{

  vtkm::Range scalar_range = m_input->GetGlobalRange(m_field_name).ReadPortal().Get(0);
  if(scalar_range.Min <= 0.f && !m_clamp_to_min)
  {
    std::stringstream msg;
    msg<<"Log : error cannot perform log on field with ";
    msg<<"negative values without clamping to a min value";
    msg<<scalar_range;
    throw Error(msg.str());
  }

  vtkm::Float32 min_value = scalar_range.Min;
  if(m_clamp_to_min)
  {
    min_value = m_min_value;
  }

  this->m_output = new DataSet();
  // shallow copy input data set and bump internal ref counts
  *m_output = *m_input;

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_output->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkm::cont::Field::Association in_assoc = dom.GetField(m_field_name).GetAssociation();
    bool is_cell_assoc = in_assoc == vtkm::cont::Field::Association::Cells;
    bool is_point_assoc = in_assoc == vtkm::cont::Field::Association::Points;

    if(!is_cell_assoc && !is_point_assoc)
    {
      throw Error("Log: input field must be zonal or nodal");
    }

    vtkm::cont::ArrayHandle<vtkm::Float32> log_field;
    vtkm::cont::Field in_field = dom.GetField(m_field_name);

    vtkm::worklet::DispatcherMapField<detail::LogField>(detail::LogField(min_value))
      .Invoke(in_field.GetData().ResetTypes(vtkm::TypeListFieldScalar(), VTKM_DEFAULT_STORAGE_LIST{}), log_field);

    vtkm::cont::Field out_field(m_result_name,
                                in_assoc,
                                log_field);
    dom.AddField(out_field);
  }
}

std::string
Log::GetName() const
{
  return "vtkh::Log";
}

Log10::Log10()
  : m_min_value(0.0001f),
    m_clamp_to_min(false)
{
}

Log10::~Log10()
{

}

void
Log10::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Log10::SetClampToMin(bool on)
{
  m_clamp_to_min = on;
}

void
Log10::SetClampMin(vtkm::Float32 min_value)
{

  if(min_value <= 0)
  {
    throw Error("Log10: min clamp value must be positive");
  }

  m_min_value = min_value;
}

void
Log10::SetResultField(const std::string &field_name)
{
  m_result_name = field_name;
}


std::string
Log10::GetField() const
{
  return m_field_name;
}

std::string
Log10::GetResultField() const
{
  return m_result_name;
}

void Log10::PreExecute()
{
  Filter::PreExecute();

  Filter::CheckForRequiredField(m_field_name);

  if(m_result_name== "")
  {
    m_result_name= "log(" + m_field_name + ")";
  }

}

void Log10::PostExecute()
{
  Filter::PostExecute();
}

void Log10::DoExecute()
{

  vtkm::Range scalar_range = m_input->GetGlobalRange(m_field_name).ReadPortal().Get(0);
  if(scalar_range.Min <= 0.f && !m_clamp_to_min)
  {
    std::stringstream msg;
    msg<<"Log10 : error cannot perform log on field with ";
    msg<<"negative values without clamping to a min value";
    msg<<scalar_range;
    throw Error(msg.str());
  }

  vtkm::Float32 min_value = scalar_range.Min;
  if(m_clamp_to_min)
  {
    min_value = m_min_value;
  }

  this->m_output = new DataSet();
  // shallow copy input data set and bump internal ref counts
  *m_output = *m_input;

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_output->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkm::cont::Field::Association in_assoc = dom.GetField(m_field_name).GetAssociation();
    bool is_cell_assoc = in_assoc == vtkm::cont::Field::Association::CELL_SET;
    bool is_point_assoc = in_assoc == vtkm::cont::Field::Association::POINTS;

    if(!is_cell_assoc && !is_point_assoc)
    {
      throw Error("Log10: input field must be zonal or nodal");
    }

    vtkm::cont::ArrayHandle<vtkm::Float32> log_field;
    vtkm::cont::Field in_field = dom.GetField(m_field_name);

    vtkm::worklet::DispatcherMapField<detail::Log10Field>(detail::Log10Field(min_value))
      .Invoke(in_field.GetData().ResetTypes(vtkm::TypeListFieldScalar(), VTKM_DEFAULT_STORAGE_LIST{}), log_field);

    vtkm::cont::Field out_field(m_result_name,
                                in_assoc,
                                log_field);
    dom.AddField(out_field);
  }
}

std::string
Log10::GetName() const
{
  return "vtkh::Log10";
}

Log2::Log2()
  : m_min_value(0.0001f),
    m_clamp_to_min(false)
{
}

Log2::~Log2()
{

}

void
Log2::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Log2::SetClampToMin(bool on)
{
  m_clamp_to_min = on;
}

void
Log2::SetClampMin(vtkm::Float32 min_value)
{

  if(min_value <= 0)
  {
    throw Error("Log2: min clamp value must be positive");
  }

  m_min_value = min_value;
}

void
Log2::SetResultField(const std::string &field_name)
{
  m_result_name = field_name;
}


std::string
Log2::GetField() const
{
  return m_field_name;
}

std::string
Log2::GetResultField() const
{
  return m_result_name;
}

void Log2::PreExecute()
{
  Filter::PreExecute();

  Filter::CheckForRequiredField(m_field_name);

  if(m_result_name== "")
  {
    m_result_name= "log(" + m_field_name + ")";
  }

}

void Log2::PostExecute()
{
  Filter::PostExecute();
}

void Log2::DoExecute()
{

  vtkm::Range scalar_range = m_input->GetGlobalRange(m_field_name).ReadPortal().Get(0);
  if(scalar_range.Min <= 0.f && !m_clamp_to_min)
  {
    std::stringstream msg;
    msg<<"Log2 : error cannot perform log on field with ";
    msg<<"negative values without clamping to a min value";
    msg<<scalar_range;
    throw Error(msg.str());
  }

  vtkm::Float32 min_value = scalar_range.Min;
  if(m_clamp_to_min)
  {
    min_value = m_min_value;
  }

  this->m_output = new DataSet();
  // shallow copy input data set and bump internal ref counts
  *m_output = *m_input;

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_output->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkm::cont::Field::Association in_assoc = dom.GetField(m_field_name).GetAssociation();
    bool is_cell_assoc = in_assoc == vtkm::cont::Field::Association::CELL_SET;
    bool is_point_assoc = in_assoc == vtkm::cont::Field::Association::POINTS;

    if(!is_cell_assoc && !is_point_assoc)
    {
      throw Error("Log2: input field must be zonal or nodal");
    }

    vtkm::cont::ArrayHandle<vtkm::Float32> log_field;
    vtkm::cont::Field in_field = dom.GetField(m_field_name);

    vtkm::worklet::DispatcherMapField<detail::Log2Field>(detail::Log2Field(min_value))
      .Invoke(in_field.GetData().ResetTypes(vtkm::TypeListFieldScalar(), VTKM_DEFAULT_STORAGE_LIST{}), log_field);

    vtkm::cont::Field out_field(m_result_name,
                                in_assoc,
                                log_field);
    dom.AddField(out_field);
  }
}

std::string
Log2::GetName() const
{
  return "vtkh::Log2";
}

} //  namespace vtkh
