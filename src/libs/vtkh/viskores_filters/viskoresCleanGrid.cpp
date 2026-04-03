#include "viskoresCleanGrid.hpp"
#include <viskores/filter/clean_grid/CleanGrid.h>

namespace vtkh
{

void
viskoresCleanGrid::tolerance(const viskores::Float64 tol)
{
  m_tolerance = tol;
}

void
viskoresCleanGrid::merge_points(bool merge) {
  m_merge_points = merge;
}

viskores::cont::DataSet
viskoresCleanGrid::Run(viskores::cont::DataSet &input,
                   viskores::filter::FieldSelection map_fields)
{
  viskores::filter::clean_grid::CleanGrid cleaner;

  if(m_tolerance != -1.)
  {
    cleaner.SetTolerance(m_tolerance);
    cleaner.SetToleranceIsAbsolute(true);
  }

  cleaner.SetFieldsToPass(map_fields);
  cleaner.SetRemoveDegenerateCells(true);
  cleaner.SetMergePoints(m_merge_points);
  auto output = cleaner.Execute(input);
  return output;
}

} // namespace vtkh
