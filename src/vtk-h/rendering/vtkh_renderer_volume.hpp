#ifndef VTK_H_RENDERER_VOLUME_HPP
#define VTK_H_RENDERER_VOLUME_HPP

#include <rendering/vtkh_renderer.hpp>

namespace vtkh {

class vtkhVolumeRenderer : public vtkhRenderer
{
public:
  vtkhVolumeRenderer();
  virtual ~vtkhVolumeRenderer();
  virtual vtkhRenderer::vtkmCanvasPtr GetNewCanvas() override;
protected:  
  virtual void SetupCanvases() override;
  virtual void Composite(const int &num_images) override;
  std::vector<std::vector<int>> m_visibility_orders;
  void FindVisibilityOrdering();
  float FindMinDepth(const vtkm::rendering::Camera &camera, 
                     const vtkm::Bounds &bounds) const;
};

} // namespace vtkh
#endif
