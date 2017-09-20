#include <rendering/vtkh_render.hpp>

namespace vtkh 
{

Render::Render()
  : m_color_table("cool2warm"),
    m_has_color_table(false)
{
}

Render::~Render()
{
}

Render::vtkmCanvasPtr 
Render::GetDomainCanvas(const vtkm::Id &domain_id)
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

Render::vtkmCanvasPtr 
Render::GetCanvas(const vtkm::Id index)
{
  assert(index >= 0 && index < m_canvases.size());
  return m_canvases[index];
}

void 
Render::AddCanvas(vtkmCanvasPtr canvas, vtkm::Id domain_id)
{
  m_canvases.push_back(canvas);
  m_domain_ids.push_back(domain_id);
}

int 
Render::GetNumberOfCanvases() const
{
  return static_cast<int>(m_canvases.size());
}

bool 
Render::HasCanvas(const vtkm::Id &domain_id) const 
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

const vtkm::rendering::Camera& 
Render::GetCamera() const
{ 
  return m_camera;
}

void 
Render::SetCamera(const vtkm::rendering::Camera &camera)
{ 
   m_camera = camera;
}

void 
Render::SetImageName(const std::string &name)
{
  m_image_name = name;
}

std::string 
Render::GetImageName() const 
{
  return m_image_name;
}

void 
Render::SetColorTable(const vtkm::rendering::ColorTable &color_table)
{
  m_color_table = color_table;
  m_has_color_table = true;
}

bool
Render::HasColorTable() const
{
  return m_has_color_table;
}

vtkm::rendering::ColorTable
Render::GetColorTable() const
{
  return m_color_table;
}

} // namespace vtkh
