#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkh/vtkh_exports.h>
#include <vtkh/Error.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/rendering/Render.hpp>
#include <vtkh/compositing/Image.hpp>

#include <viskores/rendering/Camera.h>
#include <viskores/rendering/Canvas.h>
#include <viskores/rendering/Mapper.h>

namespace vtkh {

class Compositor;

class VTKH_API Renderer : public Filter
{
public:
  typedef std::shared_ptr<viskores::rendering::Canvas> viskoresCanvasPtr;
  typedef std::shared_ptr<viskores::rendering::Mapper> viskoresMapperPtr;
  typedef viskores::rendering::Camera viskoresCamera;

  Renderer();
  virtual ~Renderer();
  virtual void SetShadingOn(bool on);
  virtual void Update();

  void AddRender(vtkh::Render &render);
  void ClearRenders();

  void SetField(const std::string field_name);
  virtual void SetColorTable(const viskores::cont::ColorTable &color_table);
  void SetDoComposite(bool do_composite);
  void SetRenders(const std::vector<Render> &renders);
  void SetRange(const viskores::Range &range);
  void SetDiscrete();
  void DisableColorBar();

  viskores::cont::ColorTable      GetColorTable() const;
  std::string                 GetFieldName() const;
  int                         GetNumberOfRenders() const;
  std::vector<Render>         GetRenders() const;
  vtkh::DataSet              *GetInput();
  viskores::Range                 GetRange() const;
  bool                        GetHasColorTable() const;
  bool                        IsDiscrete() const;
  bool                        IsMeshRenderer() const;
protected:

  // image related data with cinema support
  std::vector<vtkh::Render>                m_renders;
  int                                      m_field_index;
  Compositor                              *m_compositor;
  std::string                              m_field_name;
  bool                                     m_do_composite;
  viskoresMapperPtr                            m_mapper;
  viskores::Bounds                             m_bounds;
  viskores::Range                              m_range;
  viskores::cont::ColorTable                   m_color_table;
  bool                                     m_has_color_table;
  bool                                     m_is_discrete;
  // methods
  virtual void PreExecute() override;
  virtual void PostExecute() override;
  virtual void DoExecute() override;

  virtual void Composite(const int &num_images);
  void ImageToCanvas(Image &image, viskores::rendering::Canvas &canvas, bool get_depth);

  void RenderTiled(Render::viskoresCanvas &canvas,
                   const viskoresCamera &camera,
                   std::vector<viskores::cont::DataSet> &data_sets,
                   const viskores::Int32 tile_width,
                   const viskores::Int32 tile_height);
};

} // namespace vtkh
#endif
