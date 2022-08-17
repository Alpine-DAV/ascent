#include "vtkmThreshold.hpp"

#include <vtkm/filter/Threshold.h>
#include <vtkm/worklet/CellDeepCopy.h>


namespace vtkh
{
typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetStructured<2>>
  PermStructured2d;

typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetStructured<3>>
  PermStructured3d;

typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetExplicit<>>
  PermExplicit;

typedef  vtkm::cont::CellSetPermutation<vtkm::cont::CellSetSingleType<>>
  PermExplicitSingle;

void StripPermutation(vtkm::cont::DataSet &data_set)
{
  vtkm::cont::DynamicCellSet cell_set = data_set.GetCellSet();
  vtkm::cont::DataSet result;
  vtkm::cont::CellSetExplicit<> explicit_cells;

  if(cell_set.IsSameType(PermStructured2d()))
  {
    PermStructured2d perm = cell_set.Cast<PermStructured2d>();
    explicit_cells = vtkm::worklet::CellDeepCopy::Run(perm);
  }
  else if(cell_set.IsSameType(PermStructured3d()))
  {
    PermStructured3d perm = cell_set.Cast<PermStructured3d>();
    explicit_cells = vtkm::worklet::CellDeepCopy::Run(perm);
  }
  else if(cell_set.IsSameType(PermExplicit()))
  {
    PermExplicit perm = cell_set.Cast<PermExplicit>();
    explicit_cells = vtkm::worklet::CellDeepCopy::Run(perm);
  }
  else if(cell_set.IsSameType(PermExplicitSingle()))
  {
    PermExplicitSingle perm = cell_set.Cast<PermExplicitSingle>();
    explicit_cells = vtkm::worklet::CellDeepCopy::Run(perm);
  }

  result.SetCellSet(explicit_cells);

  vtkm::Id num_coords = data_set.GetNumberOfCoordinateSystems();
  for(vtkm::Id i = 0; i < num_coords; ++i)
  {
    result.AddCoordinateSystem(data_set.GetCoordinateSystem(i));
  }

  vtkm::Id num_fields = data_set.GetNumberOfFields();
  for(vtkm::Id i = 0; i < num_fields; ++i)
  {
    result.AddField(data_set.GetField(i));
  }

  data_set = result;
}

vtkm::cont::DataSet
vtkmThreshold::Run(vtkm::cont::DataSet &input,
                   std::string field_name,
                   double min_value,
                   double max_value,
                   vtkm::filter::FieldSelection map_fields,
                   bool return_all_in_range)
{
  vtkm::filter::Threshold thresholder;
  thresholder.SetAllInRange(return_all_in_range);
  thresholder.SetUpperThreshold(max_value);
  thresholder.SetLowerThreshold(min_value);
  thresholder.SetActiveField(field_name);
  thresholder.SetFieldsToPass(map_fields);
  auto output = thresholder.Execute(input);
  //vtkh::StripPermutation(output);
  return output;
}

} // namespace vtkh
