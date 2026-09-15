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
               viskores::cont::ArrayHandle<viskores::Float64> dirs_zs,
               double max_dist);

  void SetFields(const std::vector<std::string> &fields);

  void GenerateResultRaysMesh(const Result &result_image,
                              conduit::Node &rays_mesh);

  const viskoresCamera &GetCamera()       const { return m_camera; }
  viskores::Bounds      GetResultBounds() const { return m_bounds; }
  const Result         &GetResultImage()  const { return m_result_image; }

protected:

  int m_width;
  int m_height;
  int m_num_points;
  std::string m_mode;

  viskores::cont::ArrayHandle<viskores::Float64> m_rays_pts_xs;
  viskores::cont::ArrayHandle<viskores::Float64> m_rays_pts_ys;
  viskores::cont::ArrayHandle<viskores::Float64> m_rays_pts_zs;

  viskores::cont::ArrayHandle<viskores::Float64> m_rays_dirs_xs;
  viskores::cont::ArrayHandle<viskores::Float64> m_rays_dirs_ys;
  viskores::cont::ArrayHandle<viskores::Float64> m_rays_dirs_zs;
  double m_rays_max_distance;

  std::vector<std::string> m_field_names;

  // results
  viskoresCamera   m_camera;
  viskores::Bounds m_bounds;
  Result           m_result_image;

  // methods
  virtual void PreExecute() override;
  virtual void PostExecute() override;
  virtual void DoExecute() override;

  PayloadImage * Convert(Result &result);

  ScalarRenderer::Result Convert(PayloadImage &image, std::vector<std::string> &names);
  template <typename Precision>
  void GenerateCameraRays(const viskoresCamera &camera,
                          const viskores::Bounds &bounds,
                          int width, int height,
                          viskores::rendering::raytracing::Ray<Precision> &rays);

  template <typename Precision>
  void GenerateExplicitRays(viskores::rendering::raytracing::Ray<Precision> &rays);

  //void ImageToDataSet(Image &image, viskores::rendering::Canvas &canvas, bool get_depth);

};

} // namespace vtkh
#endif
