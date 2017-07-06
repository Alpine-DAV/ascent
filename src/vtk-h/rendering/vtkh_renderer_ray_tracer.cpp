#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <memory>

namespace vtkh {
  
vtkhRayTracer::vtkhRayTracer()
{
  typedef vtkm::rendering::MapperRayTracer TracerType;
  this->m_mapper = std::make_shared<TracerType>();
}

vtkhRayTracer::~vtkhRayTracer()
{
}

vtkhRenderer::vtkmCanvasPtr 
vtkhRayTracer::GetNewCanvas()
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>();
}

} // namespace vtkh
