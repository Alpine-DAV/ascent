#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkh_filter.hpp>

#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Canvas.h>

namespace vtkh {

class vtkhRenderer : public vtkhFilter
{
protected:
  
  // image related data with cinema support
  typedef std::shared_ptr<vtkm::rendering::Canvas> vtkmCanvasPtr; 
  typedef vtkm::rendering::Camera vtkmCamera; 
  std::vector<vtkmCanvasPtr> m_canvases;
  std::vector<vtkmCamera>    m_cameras;
  int                        m_batch_size;
  // draw annoations?? 
  bool                       m_world_annotations;   
  bool                       m_screen_annotations;   
public:
  
};

} // namespace vtkh
#endif
