#ifndef VTK_H_RENDER_HPP
#define VTK_H_RENDER_HPP

#include <vector>
#include <vtkh/vtkh_exports.h>
#include <vtkh/DataSet.hpp>
#include <vtkh/Error.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/Mapper.h>

namespace vtkh {
//
// A Render contains the information needed to create a single image.
// There are 'n' canvases that matches the number of domains in the
// data set. It is possible to chain multiple plots together that
// are rendering separate data, i.e. the result of different data
// transformations, to handle this we keep track of the domain ids
// that each canvas is associated with.
//

class VTKH_API Render
{
public:
  typedef vtkm::rendering::CanvasRayTracer vtkmCanvas;

  Render();
  ~Render();
  Render                          Copy() const;
  vtkmCanvas&                     GetCanvas();
  const vtkm::rendering::Camera&  GetCamera() const;
  std::string                     GetImageName() const;
  std::vector<std::string>        GetComments() const;
  vtkm::Bounds                    GetSceneBounds() const;
  vtkm::Int32                     GetHeight() const;
  vtkm::Int32                     GetWidth() const;
  vtkm::rendering::Color          GetBackgroundColor() const;
  bool                            GetShadingOn() const;
  void                            Print() const;

  void                            DoRenderAnnotations(bool on);
  void                            DoRenderWorldAnnotations(bool on);
  void                            DoRenderScreenAnnotations(bool on);
  void                            DoRenderBackground(bool on);
  void                            ScaleWorldAnnotations(float x, float y, float z);
  void                            SetWidth(const vtkm::Int32 width);
  void                            SetHeight(const vtkm::Int32 height);
  void                            SetSceneBounds(const vtkm::Bounds &bounds);
  void                            SetCamera(const vtkm::rendering::Camera &camera);
  void                            SetImageName(const std::string &name);
  void                            SetComments(const std::vector<std::string> &comments);
  void                            SetBackgroundColor(float bg_color[4]);
  void                            SetForegroundColor(float fg_color[4]);
  void                            SetShadingOn(bool on);
  void                            RenderWorldAnnotations();
  void                            RenderBackground();
  void                            RenderScreenAnnotations(const std::vector<std::string> &field_names,
                                                          const std::vector<vtkm::Range> &ranges,
                                                          const std::vector<vtkm::cont::ColorTable> &colors);
  void                            Save();
protected:
  vtkm::rendering::Camera      m_camera;
  std::string                  m_image_name;
  std::vector<std::string>     m_comments;
  vtkm::Bounds                 m_scene_bounds;
  vtkm::Int32                  m_width;
  vtkm::Int32                  m_height;
  vtkm::rendering::Color       m_bg_color;
  vtkm::rendering::Color       m_fg_color;
  vtkmCanvas                   CreateCanvas() const;
  bool                         m_render_annotations;
  bool                         m_render_world_annotations;
  bool                         m_render_screen_annotations;
  bool                         m_render_background;
  bool                         m_shading;
  vtkmCanvas                   m_canvas;
  vtkm::Vec<float,3>           m_world_annotation_scale;
};

static float vtkh_default_bg_color[4] = {0.f, 0.f, 0.f, 1.f};
static float vtkh_default_fg_color[4] = {1.f, 1.f, 1.f, 1.f};

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           vtkm::Bounds scene_bounds,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           vtkm::Bounds scene_bounds,
           vtkm::rendering::Camera camera,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           vtkm::rendering::Camera camera,
           vtkh::DataSet &data_set,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

} // namespace vtkh
#endif
