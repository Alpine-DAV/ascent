#include "vtkmGhostCellRemove.hpp"

#include <vtkm/filter/entity_extraction/GhostCellRemove.h>

namespace vtkh
{

vtkm::cont::DataSet
vtkmGhostStripper::Run(vtkm::cont::DataSet &input, std::string ghost_field_name)
{
  vtkm::filter::entity_extraction::GhostCellRemove ghost_buster;
  ghost_buster.SetActiveField(ghost_field_name);
  ghost_buster.RemoveAllGhost();
  auto output = ghost_buster.Execute(input);
  
  return output;
}

} // namespace vtkh
