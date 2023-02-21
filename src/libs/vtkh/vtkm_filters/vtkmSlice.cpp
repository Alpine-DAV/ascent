#include "vtkmSlice.hpp"
#include <vtkm/filter/contour/Slice.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmHistogram::Run(vtkm::cont::PartitionedDataSet &p_input,
                   std::vector<vtkm::Vec<vtkm::Float32,3>> m_points,
                   std::vector<vtkm::Vec<vtkm::Float32,3>> m_normals);
{
	//filler
	//TODO: loop through planes
	//Do we want this or is it old news?
  vtkm::Plane plane(m_points[0], m_normals[0]);
  vtkm::filter::contour::Slice slicer;

  slicer.SetImplicitFunction(plane);

  auto output = slicer.Execute(p_input);
  return output;
}

} // namespace vtkh
