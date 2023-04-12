#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>
#include <vtkh/vtkm_filters/vtkmGhostCellRemove.hpp>
#include <vtkh/vtkm_filters/vtkmThreshold.hpp>
#include <vtkh/vtkm_filters/vtkmCleanGrid.hpp>
#include <vtkh/vtkm_filters/vtkmExtractStructured.hpp>

#include <vtkm/cont/Algorithm.h>
#include <vtkm/BinaryOperators.h>

#include <limits>

namespace vtkh
{


GhostStripper::GhostStripper()
  : m_min_value(0),  // default to real zones only
    m_max_value(0)   // 0 = real, 1 = valid ghost, 2 = garbage ghost
{

}

GhostStripper::~GhostStripper()
{

}

void
GhostStripper::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
GhostStripper::SetMinValue(const vtkm::Int32 min_value)
{
  m_min_value = min_value;
}

void
GhostStripper::SetMaxValue(const vtkm::Int32 max_value)
{
  m_max_value = max_value;
}

void GhostStripper::PreExecute()
{
  Filter::PreExecute();
  if(m_min_value > m_max_value)
  {
    throw Error("GhostStripper: min_value is greater than max value.");
  }
  Filter::CheckForRequiredField(m_field_name);
}

void GhostStripper::PostExecute()
{
  Filter::PostExecute();
}

void GhostStripper::DoExecute()
{
  this->m_output = new DataSet();

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

    vtkm::cont::Field field = dom.GetField(m_field_name);
    vtkm::Range ghost_range = field.GetRange().ReadPortal().Get(0);

    if(ghost_range.Min >= m_min_value &&
    ghost_range.Max <= m_max_value)
    {
      // nothing to do here
      m_output->AddDomain(dom, domain_id);
      continue;
    }
    if(!dom.HasField(m_field_name))
    {
      m_output->AddDomain(dom,domain_id);
    }
    else
    {
      vtkh::vtkmGhostStripper stripper;
      vtkm::cont::DataSet stripper_out = stripper.Run(dom, m_field_name);
      m_output->AddDomain(stripper_out,domain_id);
    }

  }

}

std::string
GhostStripper::GetName() const
{
  return "vtkh::GhostStripper";
}

} //  namespace vtkh
