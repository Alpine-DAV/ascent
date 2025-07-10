#ifndef VTK_H_ANARI_RENDERER_HPP
#define VTK_H_ANARI_RENDERER_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/rendering/Renderer.hpp>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/interop/anari/ANARIMapperGlyphs.h>
#include <vtkm/interop/anari/ANARIMapperPoints.h>
#include <vtkm/interop/anari/ANARIMapperTriangles.h>
#include <vtkm/interop/anari/ANARIMapperVolume.h>


namespace vtkh {


class VTKH_API ANARIRenderer : public Renderer
{
public:
  ANARIRenderer();
  virtual ~ANARIRenderer();
  std::string GetName() const override;
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);
  void SetNumberOfSamples(int num_samples);
  void SetRenderers(std::vector<vtkh::ANARIRenderer*> anari_renderers);
  bool IsANARITriangle(ANARIRenderer *renderer);
  bool IsANARIVolume(ANARIRenderer *renderer);
  bool IsANARIPoint(ANARIRenderer *renderer);
  bool IsANARIGlyph(ANARIRenderer *renderer);

  void Update() override;

protected:
  virtual void PreExecute() override;
  virtual void DoExecute() override;
  virtual void PostExecute() override;

  std::shared_ptr<vtkm::rendering::MapperRayTracer> m_tracer;

  int m_num_samples;
  anari_cpp::Device m_device;
  anari_cpp::Renderer m_renderer;
  anari_cpp::Frame m_frame;
  std::vector<anari_cpp::Light> m_lights;
  std::vector<vtkh::ANARIRenderer*> m_anari_renderers;

};
// ----------------------------------------------------------------------
// Type-only subclasses for specific ANARI renderer types
// ----------------------------------------------------------------------

class VTKH_API ANARIPointRenderer : public ANARIRenderer
{
  //emtpy 
};

class VTKH_API ANARIGlyphRenderer : public ANARIRenderer
{
  //emtpy 
};

class VTKH_API ANARITriangleRenderer : public ANARIRenderer
{
  //emtpy 
};

class VTKH_API ANARIVolumeRenderer : public ANARIRenderer
{
  //emtpy 
};


} // namespace vtkh
#endif
