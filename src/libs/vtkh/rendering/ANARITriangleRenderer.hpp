#ifndef VTK_H_RENDERER_ANARI_TRIANGLE_HPP
#define VTK_H_RENDERER_ANARI_TRIANGLE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/rendering/Renderer.hpp>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/interop/anari/ANARIMapperTriangles.h>

namespace vtkh {


class VTKH_API ANARITriangleRenderer : public Renderer
{
public:
  ANARITriangleRenderer();
  virtual ~ANARITriangleRenderer();
  std::string GetName() const override;
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);

  void Update() override;

protected:
  virtual void PreExecute() override;
  virtual void DoExecute() override;
  virtual void PostExecute() override;

  void FindVisibilityOrdering();
  void DepthSort(int num_domains,
                 std::vector<float> &min_depths,
                 std::vector<int> &local_vis_order);
  float FindMinDepth(const vtkm::rendering::Camera &camera,
                     const vtkm::Bounds &bounds) const;

  std::shared_ptr<vtkm::rendering::MapperRayTracer> m_tracer;

  anari_cpp::Device m_device;
  anari_cpp::Renderer m_renderer;
  anari_cpp::Frame m_frame;
  std::vector<anari_cpp::Light> m_lights;

};

} // namespace vtkh
#endif
