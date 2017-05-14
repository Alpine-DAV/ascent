#ifndef VTK_H_RENDERER_HPP
#define VTK_H_RENDERER_HPP

#include <vector>

class vtkhRenderer
{
protected:
  typedef vtkm::rendering::Canvas vtkmCanvas; 
  typedef vtkm::rendering::Camera vtkmCamera; 
  std::vector<vtkmCanvas *> m_canvases;
  std::vector<vtkmCamera>   m_cameras;
 ; 
};

#endif
