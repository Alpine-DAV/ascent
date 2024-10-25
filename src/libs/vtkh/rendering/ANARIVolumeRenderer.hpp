#ifndef VTK_H_RENDERER_ANARI_VOLUME_HPP
#define VTK_H_RENDERER_ANARI_VOLUME_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/rendering/Renderer.hpp>
#include <vtkm/rendering/MapperVolume.h>
#include <vtkm/interop/anari/ANARIMapperVolume.h>
#include <vtkm/interop/anari/ANARIScene.h>

namespace vtkh {


class VTKH_API ANARIVolumeRenderer : public Renderer
{
public:
  ANARIVolumeRenderer();
  virtual ~ANARIVolumeRenderer();
  std::string GetName() const override;
  void SetNumberOfSamples(const int num_samples);
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);

  void Update() override;

  virtual void SetColorTable(const vtkm::cont::ColorTable &color_table) override;
protected:
  virtual void Composite(const int &num_images) override;
  virtual void PreExecute() override;
  virtual void DoExecute() override;
  virtual void PostExecute() override;

  void FindVisibilityOrdering();
  void DepthSort(int num_domains,
                 std::vector<float> &min_depths,
                 std::vector<int> &local_vis_order);
  float FindMinDepth(const vtkm::rendering::Camera &camera,
                     const vtkm::Bounds &bounds) const;

  int m_num_samples;
  float m_sample_dist;
  std::shared_ptr<vtkm::rendering::MapperVolume> m_tracer;
  vtkm::cont::ColorTable m_corrected_color_table;
  std::vector<std::vector<int>> m_visibility_orders;

  anari_cpp::Device m_device;
  anari_cpp::Renderer m_renderer;
  anari_cpp::Frame m_frame;
  std::vector<anari_cpp::Light> m_lights;

};

} // namespace vtkh
#endif
