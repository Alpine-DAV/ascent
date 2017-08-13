#include <vtkh_error.hpp>
#include <rendering/vtkh_renderer.hpp>
#include <rendering/vtkh_image.hpp>
#include <utils/vtkm_array_utils.hpp>
#include <utils/vtkm_dataset_info.hpp>
#include <utils/vtkh_png_encoder.hpp>
#include <vtkm/rendering/raytracing/Logger.h>
#ifdef PARALLEL
#include <rendering/compositing/vtkh_diy_compositor.hpp>
#endif

#include <assert.h>

namespace vtkh {

Renderer::Renderer()
  : m_color_table("cool2warm"),
    m_do_composite(true),
    m_field_index(0)
{
  m_compositor  = NULL; 
#ifdef PARALLEL
  m_compositor  = new DIYCompositor(); 
#else
  m_compositor  = new Compositor(); 
#endif

}

Renderer::~Renderer()
{
  delete m_compositor;
}

void 
Renderer::SetField(const std::string field_name)
{
  m_field_name = field_name; 
}

void 
Renderer::SetDoComposite(bool do_composite)
{
  m_do_composite = do_composite;
}

void
Renderer::AddRender(vtkh::Render &render)
{
  m_renders.push_back(render); 
}

int
Renderer::GetNumberOfRenders() const
{
  return static_cast<int>(m_renders.size());
}

void
Renderer::ClearRenders()
{
  m_renders.clear(); 
}

void Renderer::SetColorTable(const vtkm::rendering::ColorTable &color_table)
{
  m_color_table = color_table;
}

vtkm::rendering::ColorTable Renderer::GetColorTable() const
{
  return m_color_table;
}

void 
Renderer::Composite(const int &num_images)
{

  m_compositor->SetCompositeMode(Compositor::Z_BUFFER_SURFACE);
  std::cout<<"Comp "<<num_images<<"\n";
  for(int i = 0; i < num_images; ++i)
  {
    const int num_canvases = m_renders[i].GetNumberOfCanvases();
    for(int dom = 0; dom < num_canvases; ++dom)
    {
      float* color_buffer = &GetVTKMPointer(m_renders[i].GetCanvas(dom)->GetColorBuffer())[0][0]; 
      float* depth_buffer = GetVTKMPointer(m_renders[i].GetCanvas(dom)->GetDepthBuffer()); 

      int height = m_renders[i].GetCanvas(dom)->GetHeight();
      int width = m_renders[i].GetCanvas(dom)->GetWidth();

      m_compositor->AddImage(color_buffer,
                             depth_buffer,
                             width,
                             height);
    } //for dom

    Image result = m_compositor->Composite();
    const std::string image_name = m_renders[i].GetImageName() + ".png";
#ifdef PARALLEL
    if(vtkh::GetMPIRank() == 0)
    {
      result.Save(image_name);
    }
#else
    result.Save(image_name);
#endif
    m_compositor->ClearImages();
  } // for image
}

void 
Renderer::Render()
{
  if(m_mapper.get() == 0)
  {
    std::string msg = "Renderer Error: no renderer was set by sub-class"; 
    throw Error(msg);
  }

  int total_renders = static_cast<int>(m_renders.size());
  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  for(int i = 0; i < total_renders; ++i)
  {
    for(int dom = 0; dom < num_domains; ++dom)
    {
      vtkm::cont::DataSet data_set; 
      vtkm::Id domain_id;
      m_input->GetDomain(dom, data_set, domain_id);
      const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
      const vtkm::cont::Field &field = data_set.GetField(m_field_index);
      const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();
      // paint
      vtkmCanvasPtr p_canvas = m_renders[i].GetDomainCanvas(domain_id);
      const vtkmCamera &camera = m_renders[i].GetCamera();; 
      m_mapper->SetCanvas(&(*p_canvas));
      m_mapper->RenderCells(cellset,
                            coords,
                            field,
                            m_color_table,
                            camera,
                            m_range);
    }
  }

  if(m_do_composite)
  {
    this->Composite(total_renders);
  }

}
 
void 
Renderer::PreExecute() 
{
  vtkm::cont::ArrayHandle<vtkm::Range> ranges = m_input->GetGlobalRange(m_field_name);
  int num_components = ranges.GetPortalControl().GetNumberOfValues();
  //
  // current vtkm renderers only supports single component scalar fields
  //
  assert(num_components == 1);
  m_range = ranges.GetPortalControl().Get(0);
  m_bounds = m_input->GetGlobalBounds();
  m_mapper->SetActiveColorTable(m_color_table);
}

void 
Renderer::PostExecute() 
{
}

void 
Renderer::DoExecute() 
{
  Render();
}

} // namespace vtkh
