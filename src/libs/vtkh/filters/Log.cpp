#include "Log.hpp"
#include <vtkh/Error.hpp>
#include <vtkh/vtkm_filters/vtkmLog.hpp>


namespace vtkh
{

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
    m_result_name= "log_e(" + m_field_name + ")";
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

    vtkh::vtkmLog logger;
    
    auto output = logger.Run(dom,
		    	     m_field_name,
			     m_result_name,
			     in_assoc,
		   	     1,
		    	     min_value);

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
    m_result_name= "log_10(" + m_field_name + ")";
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
    bool is_cell_assoc = in_assoc == vtkm::cont::Field::Association::Cells;
    bool is_point_assoc = in_assoc == vtkm::cont::Field::Association::Points;

    if(!is_cell_assoc && !is_point_assoc)
    {
      throw Error("Log10: input field must be zonal or nodal");
    }

    vtkh::vtkmLog logger;
    
    auto output = logger.Run(dom,
		    	     m_field_name,
			     m_result_name,
			     in_assoc,
		   	     10,
		    	     min_value);
    
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
    m_result_name= "log_2(" + m_field_name + ")";
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
    bool is_cell_assoc = in_assoc == vtkm::cont::Field::Association::Cells;
    bool is_point_assoc = in_assoc == vtkm::cont::Field::Association::Points;

    if(!is_cell_assoc && !is_point_assoc)
    {
      throw Error("Log2: input field must be zonal or nodal");
    }

    vtkh::vtkmLog logger;
    
    auto output = logger.Run(dom,
		    	     m_field_name,
			     m_result_name,
			     in_assoc,
		   	     2,
		    	     min_value);
		    	     
  }
}

std::string
Log2::GetName() const
{
  return "vtkh::Log2";
}

} //  namespace vtkh
