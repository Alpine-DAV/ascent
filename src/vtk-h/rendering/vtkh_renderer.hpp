#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkh_error.hpp>
#include <vtkh_filter.hpp>
#include <rendering/compositing/vtkh_compositor.hpp>

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

  vtkmCanvasPtr GetDomainCanvas(const vtkm::Id &domain_id)
  {
    vtkm::Id dom = -1;
    for(size_t i = 0; i < m_domain_ids.size(); ++i)
    {
      if(m_domain_ids[i] == domain_id)
      {
        dom = i;
        break;
      }
    }

    if(dom == -1)
    {
      std::stringstream ss;
      ss<<"Render: canvas with domain id "<< domain_id <<" not found ";
      throw Error(ss.str());
    }
    return m_canvases[dom];
  }

  vtkmCanvasPtr GetCanvas(const vtkm::Id index)
  {
    assert(index >= 0 && index < m_canvases.size());
    return m_canvases[index];
  }

  int GetNumberOfCanvases() const
  {
    return static_cast<int>(m_canvases.size());
  }

  bool HasCanvas(const vtkm::Id &domain_id) const 
  {
    vtkm::Id dom = -1;
    for(size_t i = 0; i < m_domain_ids.size(); ++i)
    {
      if(m_domain_ids[i] == domain_id)
      {
        dom = i;
        break;
      }
    }

    return dom != -1;
  }
  
  const vtkm::rendering::Camera& GetCamera() const
  { 
    return m_camera;
  }

  void SetCamera(const vtkm::rendering::Camera &camera)
  { 
     m_camera = camera;
  }

  void SetImageName(const std::string &name)
  {
    m_image_name = name;
  }

  std::string GetImageName() const 
  {
    return m_image_name;
  }

  void AddCanvas(vtkmCanvasPtr canvas, vtkm::Id domain_id)
  {
    m_canvases.push_back(canvas);  
    m_domain_ids.push_back(domain_id);  
  }

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
  camera.Azimuth(30.f);
  camera.Elevation(25.f);
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


class Renderer : public Filter
{
public:
  typedef std::shared_ptr<vtkm::rendering::Canvas> vtkmCanvasPtr; 
  typedef std::shared_ptr<vtkm::rendering::Mapper> vtkmMapperPtr; 
  typedef vtkm::rendering::Camera vtkmCamera; 

  Renderer(); 
  virtual ~Renderer(); 
  void AddRender(vtkh::Render &render); 
  void ClearRenders(); 
  void SetField(const std::string field_name);
  void SetColorTable(const vtkm::rendering::ColorTable &color_table);
  void SetDoComposite(bool do_composite);
  vtkm::rendering::ColorTable GetColorTable() const;
  int  GetNumberOfRenders() const; 
  std::vector<Render> GetRenders(); 
  void SetRenders(const std::vector<Render> &renders);
protected:
  
  // image related data with cinema support
  std::vector<vtkh::Render>                m_renders;
  int                                      m_field_index;
  Compositor                              *m_compositor;
  std::string                              m_field_name;
  // draw annoations?? 
  bool                                     m_world_annotations;   
  bool                                     m_screen_annotations;   
  bool                                     m_do_composite;   
  vtkmMapperPtr                            m_mapper;
  vtkm::Bounds                             m_bounds;
  vtkm::Range                              m_range;
  vtkm::rendering::ColorTable              m_color_table;
  
  // methods
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute();
  //void CreateCanvases();
  virtual void Composite(const int &num_images);
};

} // namespace vtkh
#endif
