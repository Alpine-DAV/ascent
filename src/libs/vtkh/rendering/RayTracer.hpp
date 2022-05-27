#ifndef VTK_H_RENDERER_RAY_TRACER_HPP
#define VTK_H_RENDERER_RAY_TRACER_HPP

#include <vtkh/rendering/Renderer.hpp>
#include <vtkh/vtkh_exports.h>

namespace vtkh {

class VTKH_API RayTracer : public Renderer
{
public:
  RayTracer();
  virtual ~RayTracer();
  std::string GetName() const override;
  void SetShadingOn(bool on) override;
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);
};

} // namespace vtkh
#endif
