#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/vtkm_filters/vtkmGradient.hpp>

namespace vtkh
{

Gradient::Gradient()
{

}

Gradient::~Gradient()
{

}

void
Gradient::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void Gradient::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void Gradient::SetParameters(GradientParameters params)
{
  m_params = params;
}

void Gradient::PostExecute()
{
  Filter::PostExecute();
}

void Gradient::DoExecute()
{
  this->m_output = new DataSet();
  vtkh::DataSet *old_input = this->m_input;


  // make sure we have a node-centered field
  bool valid_field = false;
  bool is_cell_assoc = m_input->GetFieldAssociation(m_field_name, valid_field) ==
                       vtkm::cont::Field::Association::Cells;
  bool delete_input = false;

  if(valid_field && is_cell_assoc)
  {
    Recenter recenter;
    recenter.SetInput(m_input);
    recenter.SetField(m_field_name);
    recenter.SetResultAssoc(vtkm::cont::Field::Association::Points);
    recenter.Update();
    m_input = recenter.GetOutput();
    delete_input = true;
  }

  const int num_domains = this->m_input->GetNumberOfDomains();
  for(int i = 0; i < num_domains; ++i)
  {

    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkh::vtkmGradient grad;

    auto dataset = grad.Run(dom,
                            m_field_name,
                            m_params,
                            this->GetFieldSelection());

    m_output->AddDomain(dataset, domain_id);

  }

  if(delete_input)
  {
    delete m_input;
    this->m_input = old_input;
  }
}

std::string
Gradient::GetName() const
{
  return "vtkh::Gradient";
}

} //  namespace vtkh
