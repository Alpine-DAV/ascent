#include "viskoresGhostCellRemove.hpp"

#include <viskores/filter/entity_extraction/GhostCellRemove.h>

namespace vtkh
{

viskores::cont::DataSet
viskoresGhostStripper::Run(viskores::cont::DataSet &input, std::string ghost_field_name)
{
  input.SetGhostCellFieldName(ghost_field_name);

  viskores::filter::entity_extraction::GhostCellRemove ghost_buster;
  ghost_buster.SetActiveField(ghost_field_name);
  ghost_buster.SetTypesToRemoveToAll();
  auto output = ghost_buster.Execute(input);
  
  return output;
}

} // namespace vtkh
