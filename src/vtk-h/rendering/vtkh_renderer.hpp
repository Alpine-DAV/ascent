#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkh_filter.hpp>
#include <rendering/compositing/vtkh_compositor.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Mapper.h>

namespace vtkh {

class vtkhRenderer : public vtkhFilter
{
public:
  typedef std::shared_ptr<vtkm::rendering::Canvas> vtkmCanvasPtr; 
  typedef std::shared_ptr<vtkm::rendering::Mapper> vtkmMapperPtr; 
  typedef vtkm::rendering::Camera vtkmCamera; 

  vtkhRenderer(); 
  virtual ~vtkhRenderer(); 
  void AddCamera(const vtkm::rendering::Camera &camera); 
  void ClearCameras(); 
  void SetImageBatchSize(const int &batch_size);
  void SetField(const std::string field_name);
  void SetColorTable(const vtkm::rendering::ColorTable &color_table);
  vtkm::rendering::ColorTable GetColorTable() const;
  int  GetImageBatchSize() const;
  int  GetNumberOfCameras() const; 

protected:
  
  // image related data with cinema support
  std::vector<std::vector<vtkmCanvasPtr>>  m_canvases;
  std::vector<vtkmCamera>                  m_cameras;
  vtkmCamera                               m_default_camera;
  int                                      m_batch_size;
  int                                      m_height;
  int                                      m_width; // should we allow different image resolutions for different views?
  int                                      m_field_index;
  Compositor                              *m_compositor;
  std::string                              m_field_name;
  // draw annoations?? 
  bool                                     m_world_annotations;   
  bool                                     m_screen_annotations;   
  vtkmMapperPtr                            m_mapper;
  vtkm::Bounds                             m_bounds;
  vtkm::Range                              m_range;
  vtkm::rendering::ColorTable              m_color_table;
  float                                    m_background_color[4];
  
  // methods
  void Render();
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute();
  void SetCanvasBackgroundColor(float color[4]);
  void CreateCanvases();
  virtual void Composite(const int &num_images);
  virtual vtkmCanvasPtr GetNewCanvas() = 0;
  virtual void SetupCanvases();
};

} // namespace vtkh
#endif
