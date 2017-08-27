#ifndef VTK_H_RENDERER_MESH_HPP
#define VTK_H_RENDERER_MESH_HPP

#include <rendering/vtkh_renderer.hpp>

namespace vtkh {

class MeshRenderer : public Renderer
{
public:
  MeshRenderer();
  virtual ~MeshRenderer();
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);

  void SetIsOverlay(bool on);
  void SetShowInternal(bool on);
  bool GetIsOverlay() const;
  bool GetShowInternal() const;
protected:
  void PreExecute() override;

  bool m_is_overlay;
  bool m_show_internal;
};

} // namespace vtkh
#endif
