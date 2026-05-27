#ifndef VTK_H_SCALAR_RENDERER_HPP
#define VTK_H_SCALAR_RENDERER_HPP

#include <vector>
#include <vtkh/vtkh_exports.h>
#include <vtkh/Error.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/rendering/Render.hpp>
#include <vtkh/compositing/PayloadImage.hpp>

#include <viskores/rendering/Camera.h>
#include <viskores/rendering/ScalarRenderer.h>

#include <conduit/conduit.hpp>


namespace vtkh {

class VTKH_API ScalarRenderer : public Filter
{
public:
  typedef viskores::rendering::Camera viskoresCamera;
  using Result = viskores::rendering::ScalarRenderer::Result;

  ScalarRenderer();
  virtual ~ScalarRenderer();
  virtual void Update();
  virtual std::string GetName() const override;


  // int GetNumberOfCameras() const;
  vtkh::DataSet *GetInput();

  // camera use case
  void SetCamera(viskoresCamera &camera);
  void SetHeight(const int height);
  void SetWidth(const int width);

  // arb rays use case
  void SetRays(viskores::cont::ArrayHandle<viskores::Float64> pts_xs,
               viskores::cont::ArrayHandle<viskores::Float64> pts_ys,
               viskores::cont::ArrayHandle<viskores::Float64> pts_zs,
               viskores::cont::ArrayHandle<viskores::Float64> dirs_xs,
               viskores::cont::ArrayHandle<viskores::Float64> dirs_ys,
               viskores::cont::ArrayHandle<viskores::Float64> dirs_zs);

  void SetFields(const std::vector<std::string> &fields);

protected:

  int m_width;
  int m_height;
  int m_num_points;

  std::vector<std::string> m_field_names;

  // image related data with cinema support
  viskoresCamera  m_camera;
  // methods
  virtual void PreExecute() override;
  virtual void PostExecute() override;
  virtual void DoExecute() override;

  PayloadImage * Convert(Result &result);
  ScalarRenderer::Result Convert(PayloadImage &image, std::vector<std::string> &names);
  //void ImageToDataSet(Image &image, viskores::rendering::Canvas &canvas, bool get_depth);

};

} // namespace vtkh
#endif
