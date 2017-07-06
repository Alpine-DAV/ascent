#ifndef VTK_H_RENDERER_RAY_TRACER_HPP
#define VTK_H_RENDERER_RAY_TRACER_HPP

#include <rendering/vtkh_renderer.hpp>

namespace vtkh {

class vtkhRayTracer : public vtkhRenderer
{
public:
  vtkhRayTracer();
  virtual ~vtkhRayTracer();
  virtual vtkhRenderer::vtkmCanvasPtr GetNewCanvas() override;
};

} // namespace vtkh
#endif
