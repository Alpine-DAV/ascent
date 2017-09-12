#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkh_error.hpp>
#include <vtkh_filter.hpp>
#include <rendering/vtkh_render.hpp>
#include <rendering/compositing/vtkh_compositor.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Mapper.h>

namespace vtkh {

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
