#include "MeshRenderer.hpp"

#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperWireframer.h>
#include <memory>

namespace vtkh {

MeshRenderer::MeshRenderer()
  : m_is_overlay(false),
    m_show_internal(false),
    m_use_foreground_color(false)
{
  typedef viskores::rendering::MapperWireframer MapperType;
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

  typedef viskores::rendering::MapperWireframer MapperType;
  std::shared_ptr<MapperType> mesh_mapper =
    std::dynamic_pointer_cast<MapperType>(this->m_mapper);

  mesh_mapper->SetShowInternalZones(m_show_internal);
  mesh_mapper->SetIsOverlay(m_is_overlay);

  if(m_use_foreground_color)
  {
    viskores::rendering::Color fg = m_renders[0].GetCanvas().GetForegroundColor();
    viskores::cont::ColorTable single_color;
    viskores::Vec<viskores::Float32,3> fg_vec3_not_4;
    fg_vec3_not_4[0] = fg.Components[0];
    fg_vec3_not_4[1] = fg.Components[1];
    fg_vec3_not_4[2] = fg.Components[2];

    single_color.AddPoint(0.f, fg_vec3_not_4);
    single_color.AddPoint(1.f, fg_vec3_not_4);
    this->m_color_table = single_color;
    this->m_has_color_table = false;
  }
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

void
MeshRenderer::SetUseForegroundColor(bool on)
{
  m_use_foreground_color = on;
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

Renderer::viskoresCanvasPtr
MeshRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<viskores::rendering::CanvasRayTracer>(width, height);
}

std::string
MeshRenderer::GetName() const
{
  return "vtkh::MeshRenderer";
}

} // namespace vtkh
