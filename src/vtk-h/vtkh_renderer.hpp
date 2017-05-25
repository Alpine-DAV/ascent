#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>
#include <vtkhFilter.hpp>

class vtkhRenderer : public vtkhFilter
{
protected:
  typedef vtkm::rendering::Canvas vtkmCanvas; 
  typedef vtkm::rendering::Camera vtkmCamera; 
  std::vector<vtkmCanvas *> m_canvases;
  std::vector<vtkmCamera>   m_cameras;
  int                       m_batch_size;
   
public:
  
};

#endif
