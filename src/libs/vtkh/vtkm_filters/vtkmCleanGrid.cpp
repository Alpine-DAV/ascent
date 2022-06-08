#include "vtkmCleanGrid.hpp"
#include <vtkm/filter/CleanGrid.h>

namespace vtkh
{

void
vtkmCleanGrid::tolerance(const vtkm::Float64 tol)
{
  m_tolerance = tol;
}

vtkm::cont::DataSet
vtkmCleanGrid::Run(vtkm::cont::DataSet &input,
                   vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::CleanGrid cleaner;

  if(m_tolerance != -1.)
  {
    cleaner.SetTolerance(m_tolerance);
    cleaner.SetToleranceIsAbsolute(true);
  }

  cleaner.SetFieldsToPass(map_fields);
  cleaner.SetRemoveDegenerateCells(true);
  auto output = cleaner.Execute(input);
  return output;
}

} // namespace vtkh
