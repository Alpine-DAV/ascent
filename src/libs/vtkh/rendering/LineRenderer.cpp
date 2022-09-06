#include "LineRenderer.hpp"

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperCylinder.h>
#include <memory>

namespace vtkh {

LineRenderer::LineRenderer()
  : m_radius_set(false),
    m_radius(0.5f)
{
  typedef vtkm::rendering::MapperCylinder TracerType;
  auto mapper = std::make_shared<TracerType>();
  mapper->SetCompositeBackground(false);
  this->m_mapper = mapper;
}

LineRenderer::~LineRenderer()
{
}

Renderer::vtkmCanvasPtr
LineRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

std::string
LineRenderer::GetName() const
{
  return "vtkh::LineRenderer";
}

void
LineRenderer::SetRadius(vtkm::Float32 radius)
{
  m_radius = radius;
  m_radius_set = true;
}


void
LineRenderer::PreExecute()
{
  Renderer::PreExecute();

  typedef vtkm::rendering::MapperCylinder MapperType;
  std::shared_ptr<MapperType> mapper =
    std::dynamic_pointer_cast<MapperType>(this->m_mapper);

  // allow for the default mapper radius
  if(m_radius_set)
  {
    mapper->SetRadius(m_radius);
  }
}

} // namespace vtkh
