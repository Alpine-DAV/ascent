#include <rendering/vtkh_renderer_mesh.hpp>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperWireFramer.h>
#include <memory>

namespace vtkh {
  
MeshRenderer::MeshRenderer()
  : m_is_overlay(false),
    m_show_internal(false)
{
  typedef vtkm::rendering::MapperWireframer MapperType;
  auto mapper = std::make_shared<MapperType>();
  this->m_mapper = mapper;
}

MeshRenderer::~MeshRenderer()
{
}

void
MeshRenderer::PreExecute()
{
  Renderer::PreExecute();

  typedef vtkm::rendering::MapperWireframer MapperType;
  std::shared_ptr<MapperType> mesh_mapper = 
    std::dynamic_pointer_cast<MapperType>(this->m_mapper);

  mesh_mapper->SetShowInternalZones(m_show_internal);
  mesh_mapper->SetIsOverlay(m_is_overlay); 

  vtkm::rendering::ColorTable single_color;
  single_color.AddControlPoint(0.f, vtkm::rendering::Color::black); 
  single_color.AddControlPoint(1.f, vtkm::rendering::Color::black); 
  this->m_mapper->SetActiveColorTable(single_color);
}

void
MeshRenderer::SetIsOverlay(bool on)
{
  m_is_overlay = on;
}

void
MeshRenderer::SetShowInternal(bool on)
{
  m_is_overlay = on;
}

bool
MeshRenderer::GetIsOverlay() const
{
  return m_is_overlay;
}

bool
MeshRenderer::GetShowInternal() const
{
  return m_show_internal;
}

Renderer::vtkmCanvasPtr 
MeshRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

} // namespace vtkh
