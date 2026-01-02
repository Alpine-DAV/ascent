#ifndef VTK_H_RENDER_HPP
#define VTK_H_RENDER_HPP

#include <vector>
#include <vtkh/vtkh_exports.h>
#include <vtkh/DataSet.hpp>
#include <vtkh/Error.hpp>

#include <viskores/rendering/Camera.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/Mapper.h>

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
  typedef viskores::rendering::CanvasRayTracer viskoresCanvas;

  Render();
  ~Render();
  Render                          Copy() const;
  viskoresCanvas&                     GetCanvas();
  const viskores::rendering::Camera&  GetCamera() const;
  std::string                     GetImageName() const;
  std::vector<std::string>        GetComments() const;
  viskores::Bounds                    GetSceneBounds() const;
  viskores::Int32                     GetHeight() const;
  viskores::Int32                     GetWidth() const;
  viskores::rendering::Color          GetBackgroundColor() const;
  viskores::rendering::Color          GetForegroundColor() const;
  bool                            GetShadingOn() const;
  bool                            GetTileImage() const;
  viskores::Int32                 GetTileWidth() const;
  void                            Print() const;

  void                            DoRenderAnnotations(bool on);
  void                            DoRenderWorldAnnotations(bool on);
  void                            DoRenderScreenAnnotations(bool on);
  void                            DoRenderBackground(bool on);
  void                            ScaleWorldAnnotations(float x, float y, float z);
  void                            SetWidth(const viskores::Int32 width);
  void                            SetHeight(const viskores::Int32 height);
  void                            SetSceneBounds(const viskores::Bounds &bounds);
  void                            SetCamera(const viskores::rendering::Camera &camera);
  void                            SetImageName(const std::string &name);
  void                            SetComments(const std::vector<std::string> &comments);
  void                            SetColorBarPosition(std::vector<viskores::Bounds> color_bar_position);
  void                            SetBackgroundColor(float bg_color[4]);
  void                            SetForegroundColor(float fg_color[4]);
  void                            SetShadingOn(bool on);
  void                            SetTileImage(bool on);
  void                            SetTileWidth(const viskores::Int32 width);
  void                            RenderWorldAnnotations();
  void                            RenderBackground();
  void                            RenderScreenAnnotations(const std::vector<std::string> &field_names,
                                                          const std::vector<viskores::Range> &ranges,
                                                          const std::vector<viskores::cont::ColorTable> &colors,
                                                          const std::vector<int> &is_discrete);
  void                            Save();
protected:
  viskores::rendering::Camera      m_camera;
  std::string                  m_image_name;
  std::vector<std::string>     m_comments;
  viskores::Bounds                 m_scene_bounds;
  viskores::Int32                  m_width;
  viskores::Int32                  m_height;
  viskores::rendering::Color       m_bg_color;
  viskores::rendering::Color       m_fg_color;
  viskoresCanvas                   CreateCanvas() const;
  std::vector<viskores::Bounds>    m_color_bar_position;
  bool                         m_render_annotations;
  bool                         m_render_world_annotations;
  bool                         m_render_screen_annotations;
  bool                         m_render_background;
  bool                         m_shading;
  viskoresCanvas                   m_canvas;
  viskores::Vec<float,3>           m_world_annotation_scale;
  bool                             m_tile_image;
  viskores::Int32                  m_tile_width;
};

static float vtkh_default_bg_color[4] = {0.f, 0.f, 0.f, 1.f};
static float vtkh_default_fg_color[4] = {1.f, 1.f, 1.f, 1.f};

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           viskores::Bounds scene_bounds,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           viskores::Bounds scene_bounds,
           viskores::rendering::Camera camera,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

VTKH_API
vtkh::Render
MakeRender(int width,
           int height,
           viskores::rendering::Camera camera,
           vtkh::DataSet &data_set,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color,
           float fg_color[4] = vtkh_default_fg_color);

} // namespace vtkh
#endif
