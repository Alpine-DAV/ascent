#include <vtkh_marching_cubes.hpp>

#include <vtkm/filter/MarchingCubes.h>

namespace vtkh 
{

vtkhMarchingCubes::vtkhMarchingCubes()
{

}

vtkhMarchingCubes::~vtkhMarchingCubes()
{

}


void 
vtkhMarchingCubes::SetIsoValue(const double &iso_value)
{
  m_iso_values.clear();
  m_iso_values.push_back(iso_value);
}

void 
vtkhMarchingCubes::SetIsoValues(const double *iso_values, const int &num_values)
{
  assert(num_values > 0);
  m_iso_values.clear();
  for(int i = 0; i < num_values; ++i)
  {
    m_iso_values.push_back(iso_values[i]);
  }
}

void 
vtkhMarchingCubes::AddMapField(const std::string &field_name)
{
  m_map_fields.push_back(field_name);
}

void 
vtkhMarchingCubes::ClearMapFields()
{
  m_map_fields.clear();
}

void 
vtkhMarchingCubes::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}
void vtkhMarchingCubes::PreExecute() 
{
  assert(m_iso_values.size() > 0);
  assert(m_field_name != "");
  if(m_map_fields.size() == 0)
  {
    m_map_fields.push_back(m_field_name);
  }
}

void vtkhMarchingCubes::PostExecute()
{

}

bool 
vtkhMarchingCubes::ContainsIsoValues(vtkm::cont::DataSet &dom)
{
  vtkm::cont::Field field = dom.GetField(m_field_name);
  vtkm::cont::ArrayHandle<vtkm::Range> ranges = field.GetRange();
  assert(ranges.GetNumberOfValues() == 1);
  vtkm::Range range = ranges.GetPortalControl().Get(0);
  for(size_t i = 0; i < m_iso_values.size(); ++i)
  {
    double val = m_iso_values[i];
    if(range.Contains(val))
    {
      return true;
    }
  }

  return false;

}

void vtkhMarchingCubes::DoExecute()
{
  this->m_output = new vtkhDataSet();
  vtkm::filter::MarchingCubes marcher;

  marcher.SetIsoValues(m_iso_values);
 
  const int num_domains = this->m_input->GetNumberOfDomains(); 

  for(int i = 0; i < num_domains; ++i)
  {
    int domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    bool valid_domain = ContainsIsoValues(dom);
    if(!valid_domain)
    {
      // vtkm does not like it if we ask it to contour
      // values that do not exist in the field, so
      // we have to check.
      continue;
    }
    vtkm::filter::ResultDataSet res = marcher.Execute(dom, m_field_name);
    for(size_t f = 0; f < m_map_fields.size(); ++f)
    {
      marcher.MapFieldOntoOutput(res, dom.GetField(m_map_fields[f]));
    }
    this->m_output->AddDomain(res.GetDataSet(), domain_id);
    
  }
}

} //  namespace vtkh
