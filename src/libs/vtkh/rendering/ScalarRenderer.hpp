#ifndef VTK_H_SCALAR_RENDERER_HPP
#define VTK_H_SCALAR_RENDERER_HPP

#include <vector>
#include <vtkh/vtkh_exports.h>
#include <vtkh/Error.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/rendering/Render.hpp>
#include <vtkh/compositing/PayloadImage.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/ScalarRenderer.h>

namespace vtkh {

class VTKH_API ScalarRenderer : public Filter
{
public:
  typedef vtkm::rendering::Camera vtkmCamera;
  using Result = vtkm::rendering::ScalarRenderer::Result;

  ScalarRenderer();
  virtual ~ScalarRenderer();
  virtual void Update();
  virtual std::string GetName() const override;

  void SetCamera(vtkmCamera &camera);

  int GetNumberOfCameras() const;
  vtkh::DataSet *GetInput();
  void SetHeight(const int height);
  void SetWidth(const int width);
protected:

  int m_width;
  int m_height;
  // image related data with cinema support
  vtkmCamera  m_camera;
  // methods
  virtual void PreExecute() override;
  virtual void PostExecute() override;
  virtual void DoExecute() override;

  PayloadImage * Convert(Result &result);
  ScalarRenderer::Result Convert(PayloadImage &image, std::vector<std::string> &names);
  //void ImageToDataSet(Image &image, vtkm::rendering::Canvas &canvas, bool get_depth);
};

} // namespace vtkh
#endif
