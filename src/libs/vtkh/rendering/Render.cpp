#include "Render.hpp"
#include <vtkh/rendering/Annotator.hpp>
#include <vtkh/utils/PNGEncoder.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/View2D.h>
#include <vtkm/rendering/View3D.h>

namespace vtkh
{

Render::Render()
  : m_width(1024),
    m_height(1024),
    m_render_annotations(true),
    m_render_world_annotations(true),
    m_render_screen_annotations(true),
    m_render_background(true),
    m_shading(true),
    m_canvas(m_width, m_height)
{
  m_world_annotation_scale[0] = 1.f;
  m_world_annotation_scale[1] = 1.f;
  m_world_annotation_scale[2] = 1.f;
}

Render::~Render()
{
}

Render::vtkmCanvas&
Render::GetCanvas()
{
  return m_canvas;
}

vtkm::Bounds
Render::GetSceneBounds() const
{
  return m_scene_bounds;
}

void
Render::DoRenderAnnotations(bool on)
{
  m_render_annotations = on;
}

void
Render::DoRenderWorldAnnotations(bool on)
{
  m_render_world_annotations = on;
}


void
Render::DoRenderScreenAnnotations(bool on)
{
  m_render_screen_annotations = on;
}


void
Render::DoRenderBackground(bool on)
{
  m_render_background = on;
}

void
Render::ScaleWorldAnnotations(float x, float y, float z)
{
  m_world_annotation_scale[0] = x;
  m_world_annotation_scale[1] = y;
  m_world_annotation_scale[2] = z;
}

vtkm::Int32
Render::GetWidth() const
{
  return m_width;
}

vtkm::Int32
Render::GetHeight() const
{
  return m_height;
}

void
Render::SetWidth(const vtkm::Int32 width)
{
  if(width == m_width) return;
  m_width = width;
  m_canvas.ResizeBuffers(m_width, m_height);
}

void
Render::SetShadingOn(bool on)
{
  m_shading = on;
}

bool
Render::GetShadingOn() const
{
  return m_shading;
}

void
Render::SetHeight(const vtkm::Int32 height)
{
  if(height == m_height) return;
  m_height = height;
  m_canvas.ResizeBuffers(m_width, m_height);
}

void
Render::SetSceneBounds(const vtkm::Bounds &bounds)
{
  m_scene_bounds = bounds;
}

const vtkm::rendering::Camera&
Render::GetCamera() const
{
  return m_camera;
}

void
Render::SetCamera(const vtkm::rendering::Camera &camera)
{
   m_camera = camera;
}

void
Render::SetImageName(const std::string &name)
{
  m_image_name = name;
}

void
Render::SetComments(const std::vector<std::string> &comments)
{
  m_comments = comments;
}

void
Render::SetBackgroundColor(float bg_color[4])
{
  m_bg_color.Components[0] = bg_color[0];
  m_bg_color.Components[1] = bg_color[1];
  m_bg_color.Components[2] = bg_color[2];
  m_bg_color.Components[3] = bg_color[3];
}

void
Render::SetForegroundColor(float fg_color[4])
{
  m_fg_color.Components[0] = fg_color[0];
  m_fg_color.Components[1] = fg_color[1];
  m_fg_color.Components[2] = fg_color[2];
  m_fg_color.Components[3] = fg_color[3];
}

std::string
Render::GetImageName() const
{
  return m_image_name;
}

std::vector<std::string>
Render::GetComments() const
{
  return m_comments;
}

vtkm::rendering::Color
Render::GetBackgroundColor() const
{
  return m_bg_color;
}

void
Render::RenderWorldAnnotations()
{
  if(!m_render_annotations) return;
  if(!m_render_world_annotations) return;
#ifdef VTKH_PARALLEL
  if(vtkh::GetMPIRank() != 0) return;
#endif
  m_canvas.SetBackgroundColor(m_bg_color);
  m_canvas.SetForegroundColor(m_fg_color);

  Annotator annotator(m_canvas, m_camera, m_scene_bounds);
  annotator.RenderWorldAnnotations(m_world_annotation_scale);

}

void
Render::RenderScreenAnnotations(const std::vector<std::string> &field_names,
                                const std::vector<vtkm::Range> &ranges,
                                const std::vector<vtkm::cont::ColorTable> &colors)
{
  if(!m_render_annotations) return;
  if(!m_render_screen_annotations) return;
#ifdef VTKH_PARALLEL
  if(vtkh::GetMPIRank() != 0) return;
#endif
  m_canvas.SetBackgroundColor(m_bg_color);
  m_canvas.SetForegroundColor(m_fg_color);
  if(m_render_background) m_canvas.BlendBackground();

  if(!m_render_annotations) return;
  Annotator annotator(m_canvas, m_camera, m_scene_bounds);
  annotator.RenderScreenAnnotations(field_names, ranges, colors);
}

Render
Render::Copy() const
{
  Render copy;
  copy.m_camera = m_camera;
  copy.m_image_name = m_image_name;
  copy.m_scene_bounds = m_scene_bounds;
  copy.m_width = m_width;
  copy.m_height = m_height;
  copy.m_bg_color = m_bg_color;
  copy.m_fg_color = m_fg_color;
  copy.m_render_annotations = m_render_annotations;
  copy.m_render_background = m_render_background;
  copy.m_shading = m_shading;
  copy.m_canvas = CreateCanvas();
  copy.m_world_annotation_scale = m_world_annotation_scale;
  return copy;
}

void
Render::Print() const
{
  std::cout<<"=== image name  : "<<m_image_name<<"\n";;
  std::cout<<"=== bounds .... : "<<m_scene_bounds<<"\n";
  std::cout<<"=== width ..... : "<<m_width<<"\n";
  std::cout<<"=== height .... : "<<m_height<<"\n";
  std::cout<<"=== bg_color .. : "
            <<m_bg_color.Components[0]<<" "
            <<m_bg_color.Components[1]<<" "
            <<m_bg_color.Components[2]<<" "
            <<m_bg_color.Components[3]<<"\n";
  std::cout<<"=== fg_color .. : "
            <<m_fg_color.Components[0]<<" "
            <<m_fg_color.Components[1]<<" "
            <<m_fg_color.Components[2]<<" "
            <<m_fg_color.Components[3]<<"\n";
  std::cout<<"=== annotations : "
           <<(m_render_annotations ? "On" : "Off")
           <<"\n";
  std::cout<<"=== background  : "
           <<(m_render_background ? "On" : "Off")
           <<"\n";
  std::cout<<"=== shading ... : "
           <<(m_shading ? "On" : "Off")
           <<"\n";
}

void
Render::RenderBackground()
{
  if(m_render_background) m_canvas.BlendBackground();
}

Render::vtkmCanvas
Render::CreateCanvas() const
{
  Render::vtkmCanvas canvas(m_width, m_height);
  canvas.SetBackgroundColor(m_bg_color);
  canvas.SetForegroundColor(m_fg_color);
  canvas.Clear();
  return canvas;
}

void
Render::Save()
{
  // After rendering and compositing
  // Rank 0 contains the complete image.
#ifdef VTKH_PARALLEL
  if(vtkh::GetMPIRank() != 0) return;
#endif
  float* color_buffer = &GetVTKMPointer(m_canvas.GetColorBuffer())[0][0];
  int height = m_canvas.GetHeight();
  int width = m_canvas.GetWidth();
  PNGEncoder encoder;
  encoder.Encode(color_buffer, width, height, m_comments);
  encoder.Save(m_image_name + ".png");
}

vtkh::Render
MakeRender(int width,
           int height,
           vtkm::Bounds scene_bounds,
           const std::string &image_name,
           float bg_color[4],
           float fg_color[4])
{
  vtkh::Render render;
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(scene_bounds);
  render.SetSceneBounds(scene_bounds);
  render.SetWidth(width);
  render.SetHeight(height);
  //
  // detect a 2d data set
  //
  camera.SetModeTo3D();

  bool is_2d = scene_bounds.Z.Min == 0. && scene_bounds.Z.Max == 0.;

  if(is_2d)
  {
    camera.SetModeTo2D();
    render.SetShadingOn(false);
  }

  render.SetCamera(camera);
  render.SetImageName(image_name);
  render.SetBackgroundColor(bg_color);
  render.SetForegroundColor(fg_color);

  return render;
}

vtkh::Render
MakeRender(int width,
           int height,
           vtkm::Bounds scene_bounds,
           vtkm::rendering::Camera camera,
           const std::string &image_name,
           float bg_color[4],
           float fg_color[4])
{
  vtkh::Render render;
  render.SetSceneBounds(scene_bounds);
  render.SetWidth(width);
  render.SetHeight(height);
  //
  // detect a 2d data set
  //
  camera.SetModeTo3D();

  bool is_2d = scene_bounds.Z.Min == 0. && scene_bounds.Z.Max == 0.;

  if(is_2d)
  {
    camera.SetModeTo2D();
    render.SetShadingOn(false);
  }

  render.SetCamera(camera);
  render.SetImageName(image_name);
  render.SetBackgroundColor(bg_color);
  render.SetForegroundColor(fg_color);

  return render;
}

vtkh::Render
MakeRender(int width,
           int height,
           vtkm::rendering::Camera camera,
           vtkm::cont::PartitionedDataSet &data_set,
           const std::string &image_name,
           float bg_color[4],
           float fg_color[4])
{
  vtkh::Render render;
  render.SetCamera(camera);
  render.SetImageName(image_name);
  vtkm::Bounds bounds = GetGlobalBounds(&data_set);
  render.SetSceneBounds(bounds);
  render.SetWidth(width);
  render.SetHeight(height);
  //
  // detect a 2d data set
  //
  bool is_2d = bounds.Z.Min == 0. && bounds.Z.Max == 0.;

  if(is_2d)
  {
    camera.SetModeTo2D();
    render.SetShadingOn(false);
  }

  render.SetBackgroundColor(bg_color);
  render.SetForegroundColor(fg_color);

  return render;
}
} // namespace vtkh
