#include "vtkmSlice.hpp"
#include <vtkm/filter/contour/Slice.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmSlice::Run(vtkm::cont::DataSet &p_input,
                   vtkm::Vec<vtkm::Float32,3> m_point,
                   vtkm::Vec<vtkm::Float32,3> m_normal,
		   std::string fname)
{
	//filler
	//TODO: loop through planes
	//Do we want this or is it old news?
  vtkm::Plane plane(m_point, m_normal);
  vtkm::filter::contour::Slice slicer;

  slicer.SetOutputFieldName(fname);
  slicer.SetImplicitFunction(plane);

  auto output = slicer.Execute(p_input);
  return output;
}

} // namespace vtkh
