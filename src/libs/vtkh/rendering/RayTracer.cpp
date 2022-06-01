#include "RayTracer.hpp"

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <memory>

namespace vtkh {
  
RayTracer::RayTracer()
{
  typedef vtkm::rendering::MapperRayTracer TracerType;
  auto mapper = std::make_shared<TracerType>();
  mapper->SetCompositeBackground(false);
  this->m_mapper = mapper;
}

RayTracer::~RayTracer()
{
}

Renderer::vtkmCanvasPtr 
RayTracer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

std::string
RayTracer::GetName() const
{
  return "vtkh::RayTracer";
}

void 
RayTracer::SetShadingOn(bool on)
{
  // do nothing by default;
  typedef vtkm::rendering::MapperRayTracer TracerType;
  std::static_pointer_cast<TracerType>(this->m_mapper)->SetShadingOn(on);
}

} // namespace vtkh
