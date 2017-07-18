#ifndef VTK_H_RENDERER_VOLUME_HPP
#define VTK_H_RENDERER_VOLUME_HPP

#include <rendering/vtkh_renderer.hpp>

#include <vtkm/rendering/MapperVolume.h>

namespace vtkh {

class VolumeRenderer : public Renderer
{
public:
  VolumeRenderer();
  virtual ~VolumeRenderer();
  virtual Renderer::vtkmCanvasPtr GetNewCanvas() override;
  void SetNumberOfSamples(const int num_samples);
protected:  
  virtual void SetupCanvases() override;
  virtual void Composite(const int &num_images) override;
  void PreExecute() override;

  std::vector<std::vector<int>> m_visibility_orders;
  void FindVisibilityOrdering();
  void DepthSort(const int &num_domains, 
                 const std::vector<float> &min_depths,
                 std::vector<int> &local_vis_order);
  float FindMinDepth(const vtkm::rendering::Camera &camera, 
                     const vtkm::Bounds &bounds) const;
  
  std::shared_ptr<vtkm::rendering::MapperVolume> m_tracer;
  int m_num_samples;
};

} // namespace vtkh
#endif
