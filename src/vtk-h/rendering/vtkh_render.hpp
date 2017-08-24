#ifndef VTK_H_RENDER_HPP
#define VTK_H_RENDER_HPP

#include <vector>
#include <vtkh_data_set.hpp>
#include <vtkh_error.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Mapper.h>

namespace vtkh {
//
// A Render contains the information needed to create a single image.
// There are 'n' canvases that matches the number of domains in the 
// data set. It is possible to chain multiple plots together that 
// is rendering separate data, i.e. the result of different data
// transformations, to handle this we keep track of the domain ids
// that each canvas is associated with.
//

class Render
{
public: 
  typedef std::shared_ptr<vtkm::rendering::Canvas> vtkmCanvasPtr; 

  Render();
  ~Render();
  vtkmCanvasPtr                   GetDomainCanvas(const vtkm::Id &domain_id);
  vtkmCanvasPtr                   GetCanvas(const vtkm::Id index);
  int                             GetNumberOfCanvases() const;
  const vtkm::rendering::Camera&  GetCamera() const;
  std::string                     GetImageName() const;
  void                            SetCamera(const vtkm::rendering::Camera &camera);
  void                            SetImageName(const std::string &name);
  bool                            HasCanvas(const vtkm::Id &domain_id) const;
  void                            AddCanvas(vtkmCanvasPtr canvas, vtkm::Id domain_id);

protected:
  std::vector<vtkmCanvasPtr> m_canvases;
  std::vector<vtkm::Id>      m_domain_ids;
  vtkm::rendering::Camera    m_camera; 
  std::string                m_image_name;
}; 

static float vtkh_default_bg_color[4] = {1.f, 1.f, 1.f, 1.f};

template<typename RendererType>
vtkh::Render 
MakeRender(int width,
           int height, 
           vtkm::Bounds scene_bounds,
           const std::vector<vtkm::Id> &domain_ids,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color)
{
  vtkh::Render render;
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(scene_bounds);
  //
  // detect a 2d data set
  //
  float total_extent[3];
  total_extent[0] = scene_bounds.X.Length();
  total_extent[1] = scene_bounds.Y.Length();
  total_extent[2] = scene_bounds.Z.Length();
  int min_dim = 0;
  if(total_extent[1] < total_extent[min_dim]) min_dim = 1;
  if(total_extent[2] < total_extent[min_dim]) min_dim = 2;
  
  bool is_2d = (total_extent[min_dim] == 0.f);
  if(!is_2d)
  {
    camera.Azimuth(30.f);
    camera.Elevation(25.f);
  }

  render.SetCamera(camera);
  render.SetImageName(image_name);
  
  vtkm::rendering::Color color;
  color.Components[0] = bg_color[0];
  color.Components[1] = bg_color[1];
  color.Components[2] = bg_color[2];
  color.Components[3] = bg_color[3];

  for(size_t i = 0; i < domain_ids.size(); ++i)
  {
    auto canvas = RendererType::GetNewCanvas(width, height);
    canvas->Clear();
    canvas->SetBackgroundColor(color);
    render.AddCanvas(canvas, domain_ids[i]);
  }
  return render;
}

template<typename RendererType>
vtkh::Render 
MakeRender(int width,
           int height, 
           vtkm::rendering::Camera camera,
           vtkh::DataSet &data_set,
           const std::string &image_name,
           float bg_color[4] = vtkh_default_bg_color)
{
  vtkh::Render render;
  render.SetCamera(camera);
  render.SetImageName(image_name);

  vtkm::rendering::Color color;
  color.Components[0] = bg_color[0];
  color.Components[1] = bg_color[1];
  color.Components[2] = bg_color[2];
  color.Components[3] = bg_color[3];

  int num_domains = static_cast<int>(data_set.GetNumberOfDomains());
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet ds; 
    vtkm::Id domain_id;
    data_set.GetDomain(i, ds, domain_id);
    auto canvas = RendererType::GetNewCanvas(width, height);

    canvas->SetBackgroundColor(color);
    canvas->Clear();

    render.AddCanvas(canvas, domain_id);
  }
  return render;
}

} // namespace vtkh
#endif
