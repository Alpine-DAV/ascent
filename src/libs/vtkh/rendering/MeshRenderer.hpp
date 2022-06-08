#ifndef VTK_H_RENDERER_MESH_HPP
#define VTK_H_RENDERER_MESH_HPP

#include <vtkh/rendering/Renderer.hpp>
#include<vtkh/vtkh_exports.h>

namespace vtkh {

class VTKH_API MeshRenderer : public Renderer
{
public:
  MeshRenderer();
  virtual ~MeshRenderer();
  std::string GetName() const override;
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);

  void SetIsOverlay(bool on);
  void SetShowInternal(bool on);
  void SetUseForegroundColor(bool on);
  bool GetIsOverlay() const;
  bool GetShowInternal() const;
protected:
  void PreExecute() override;
  bool m_use_foreground_color;
  bool m_is_overlay;
  bool m_show_internal;
};

} // namespace vtkh
#endif
