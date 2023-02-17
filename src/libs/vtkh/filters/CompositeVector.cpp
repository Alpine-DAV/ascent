//#include <vtkm/filter/your_vtkm_filter.h>
#include <vtkh/filters/CompositeVector.hpp>
#include <vtkh/vtkm_filters/vtkmCompositeVector.hpp>
#include <vtkh/Error.hpp>

namespace vtkh
{

namespace detail
{

std::string to_string(vtkm::cont::Field::Association assoc)
{
  std::string res = "unknown";
  if(assoc == vtkm::cont::Field::Association::WholeDataSet)
  {
    res = "whole mesh";
  }
  else if(assoc == vtkm::cont::Field::Association::Any)
  {
    res = "any";
  }
  else if(assoc == vtkm::cont::Field::Association::Points)
  {
    res = "points";
  }
  else if(assoc == vtkm::cont::Field::Association::Cells)
  {
    res = "cell set";
  }
  return res;
}

}// namespace detail

CompositeVector::CompositeVector()
{

}

CompositeVector::~CompositeVector()
{

}

void
CompositeVector::SetFields(const std::string &field1,
                    const std::string &field2,
                    const std::string &field3)
{
  m_field_1 = field1;
  m_field_2 = field2;
  m_field_3 = field3;
  m_mode_3d = true;
}

void
CompositeVector::SetFields(const std::string &field1,
                    const std::string &field2)
{
  m_field_1 = field1;
  m_field_2 = field2;
  m_mode_3d = false;
}

void
CompositeVector::SetResultField(const std::string &result_name)
{
  m_result_name = result_name;
}

void CompositeVector::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_1);
  Filter::CheckForRequiredField(m_field_2);
  if(m_mode_3d)
  {
    Filter::CheckForRequiredField(m_field_3);
  }

  vtkm::Id field_1_comps = this->m_input->NumberOfComponents(m_field_1);
  vtkm::Id field_2_comps = this->m_input->NumberOfComponents(m_field_2);

  vtkm::Id min_comps = std::min(field_1_comps,field_2_comps);
  vtkm::Id max_comps = std::max(field_1_comps, field_2_comps);

  vtkm::Id field_3_comps;
  if(m_mode_3d)
  {
    field_3_comps = this->m_input->NumberOfComponents(m_field_3);
    min_comps = std::min(min_comps, field_3_comps);
    max_comps = std::max(max_comps, field_3_comps);
  }

  if((min_comps != 1) || (min_comps != max_comps))
  {
    std::stringstream ss;
    ss<<"CompositeVector: all fields need to be scalars. ";
    ss<<"'"<<m_field_1<<"' has "<<field_1_comps<<". ";
    ss<<"'"<<m_field_2<<"' has "<<field_2_comps<<". ";
    if(m_mode_3d)
    {
      ss<<"'"<<m_field_3<<"' has "<<field_3_comps<<". ";
    }
    throw Error(ss.str());
  }

  bool valid;
  vtkm::cont::Field::Association assoc_1 =
    this->m_input->GetFieldAssociation(m_field_1, valid);

  vtkm::cont::Field::Association assoc_2 =
    this->m_input->GetFieldAssociation(m_field_2, valid);

  vtkm::cont::Field::Association assoc_3;
  bool same_assoc = (assoc_1 == assoc_2);
  if(m_mode_3d)
  {
    assoc_3 = this->m_input->GetFieldAssociation(m_field_3, valid);
    same_assoc &= assoc_1 == assoc_3;
  }


  if(!same_assoc)
  {
    std::stringstream ss;
    ss<<"CompositeVector: all fields need to have same associations. ";
    ss<<"'"<<m_field_1<<"' is "<<detail::to_string(assoc_1)<<". ";
    ss<<"'"<<m_field_2<<"' is "<<detail::to_string(assoc_2)<<". ";
    if(m_mode_3d)
    {
      ss<<"'"<<m_field_3<<"' is "<<detail::to_string(assoc_3)<<". ";
    }
    throw Error(ss.str());
  }

  if(m_result_name == "")
  {
    throw Error("Vector: result name never set");
  }
}

void CompositeVector::PostExecute()
{
  Filter::PostExecute();
}

void CompositeVector::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();
  std::vector<vtkm::Id> domain_ids = this->m_input->GetDomainIds();

  bool valid;
  vtkm::cont::Field::Association assoc =
    this->m_input->GetFieldAssociation(m_field_1, valid);

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_input->GetDomain(i);
    std::vector<std::string> input_field_names;
    if(!dom.HasField(m_field_1))
    {
      continue;
    }

    input_field_names.push_back(m_field_1);
    input_field_names.push_back(m_field_2);

    if(m_mode_3d)
    {
      input_field_names.push_back(m_field_3);
      vtkmCompositeVector composite3DVec;
      vtkm::cont::DataSet output = composite3DVec.Run(dom, input_field_names, m_result_name, assoc);
      m_output->AddDomain(output, domain_ids[i]);
    }
    else
    {
      vtkmCompositeVector composite2DVec;
      vtkm::cont::DataSet output = composite2DVec.Run(dom, input_field_names, m_result_name, assoc); 
      m_output->AddDomain(output, domain_ids[i]);
    }
  }
}

std::string
CompositeVector::GetName() const
{
  return "vtkh::CompositeVector";
}

} //  namespace vtkh
