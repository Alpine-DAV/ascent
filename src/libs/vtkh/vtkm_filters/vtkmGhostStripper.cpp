#include "vtkmGhostStripper.hpp"

#include <vtkm/filter/entity_extraction/GhostCellRemove.h>

namespace vtkh
{

vtkm::cont::DataSet
vtkmGhostStripper::Run(vtkm::cont::DataSet &input, std::string ghost_field_name)
{
  vtkm::filter::entity_extraction::GhostCellRemove stripper;
  stripper.SetActiveField(ghost_field_name);
  stripper.RemoveGhostField();
  auto output = stripper.Execute(input);
  
  return output;
}

} // namespace vtkh
