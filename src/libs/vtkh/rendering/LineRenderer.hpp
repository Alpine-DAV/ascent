#ifndef VTK_H_RENDERER_LINES_HPP
#define VTK_H_RENDERER_LINES_HPP

#include <vtkh/rendering/Renderer.hpp>
#include<vtkh/vtkh_exports.h>

namespace vtkh {

class VTKH_API LineRenderer : public Renderer
{
public:
  LineRenderer();
  virtual ~LineRenderer();
  std::string GetName() const override;
  static Renderer::viskoresCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);
  void PreExecute() override;
  void SetRadius(viskores::Float32 radius);
private:
  bool m_radius_set;
  viskores::Float32 m_radius;

};

} // namespace vtkh
#endif
