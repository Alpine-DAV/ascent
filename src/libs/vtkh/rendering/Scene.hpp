#ifndef VTKH_SCENE_HPP
#define VTKH_SCENE_HPP

#include <vector>
#include <list>
#include <vtkh/vtkh_exports.h>
#include <vtkh/rendering/Render.hpp>
#include <vtkh/rendering/Renderer.hpp>

namespace vtkh
{

class VTKH_API Scene
{
private:
  std::list<vtkh::Renderer*>   m_renderers;
  std::vector<vtkh::Render>    m_renders;
  bool                         m_has_volume;
  int                          m_batch_size;
public:
 Scene();
 ~Scene();

  void AddRender(vtkh::Render &render);
  void SetRenders(const std::vector<vtkh::Render> &renders);
  void AddRenderer(vtkh::Renderer *render);
  void Render();
  void Save();
  void SetRenderBatchSize(int batch_size);
  int  GetRenderBatchSize() const;
protected:
  bool IsMesh(vtkh::Renderer *renderer);
  bool IsVolume(vtkh::Renderer *renderer);
  void SynchDepths(std::vector<vtkh::Render> &renders);
}; // class scene

} //namespace  vtkh
#endif
