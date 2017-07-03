#include <vtkh_error.hpp>
#include <rendering/vtkh_renderer.hpp>
#include <rendering/vtkh_image.hpp>
#include <utils/vtkm_array_utils.hpp>
#include <utils/vtkm_dataset_info.hpp>
#include <utils/vtkh_png_encoder.hpp>

#ifdef PARALLEL
#include <rendering/compositing/vtkh_diy_compositor.hpp>
#endif

#include <assert.h>

namespace vtkh {

vtkhRenderer::vtkhRenderer()
  : m_color_table("cool2warm"),
    m_width(1024),
    m_height(1024),
    m_batch_size(10),
    m_field_index(0),
    m_background_color{1.f, 1.f, 1.f, 1.f}
{
  m_compositor  = NULL; 
#ifdef PARALLEL
  m_compositor  = new DIYCompositor(); 
#else
  m_compositor  = new Compositor(); 
#endif

}

vtkhRenderer::~vtkhRenderer()
{
  delete m_compositor;
}

void 
vtkhRenderer::SetField(const std::string field_name)
{
  m_field_name = field_name; 
}

void
vtkhRenderer::AddCamera(const vtkm::rendering::Camera &camera)
{
  m_cameras.push_back(camera); 
}

int
vtkhRenderer::GetNumberOfCameras() const
{
  return static_cast<int>(m_cameras.size());
}

void
vtkhRenderer::ClearCameras()
{
  m_cameras.clear(); 
}

void 
vtkhRenderer::SetImageBatchSize(const int &batch_size)
{
  assert(batch_size > 0);
  m_batch_size = batch_size;
}

int
vtkhRenderer::GetImageBatchSize() const 
{
  return m_batch_size;
}

void vtkhRenderer::SetColorTable(const vtkm::rendering::ColorTable &color_table)
{
  m_color_table = color_table;
}

vtkm::rendering::ColorTable vtkhRenderer::GetColorTable() const
{
  return m_color_table;
}

void 
vtkhRenderer::CreateCanvases()
{
  int num_cameras = static_cast<int>(m_cameras.size()); 
  int num_canvases = std::min(m_batch_size, num_cameras);
  
  int current_size = static_cast<int>(m_canvases.size());
  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  m_canvases.resize(num_domains);
  
  while(current_size < num_canvases)
  {
    for(int i = 0; i < num_domains; ++i)
    {
      //TODO: this could change if we can pass the same canvas to
      //      vtkm renderers
      //      Alternative: just take care of this when we transfer to
      //      unsigned char
      m_canvases[i].push_back(this->GetNewCanvas()); 
    }
    current_size++;
  }
  
}

void 
vtkhRenderer::SetupCanvases()
{
  CreateCanvases();  
  SetCanvasBackgroundColor(m_background_color);
}


void 
vtkhRenderer::SetCanvasBackgroundColor(float color[4])
{
  int current_size = static_cast<int>(m_canvases.size());
  vtkm::rendering::Color vtkm_color;
  vtkm_color.Components[0] = color[0];
  vtkm_color.Components[1] = color[1];
  vtkm_color.Components[2] = color[2];
  vtkm_color.Components[3] = color[3];
  for(int i = 0; i < current_size; ++i)
  {
    int num_canvases = static_cast<int>(m_canvases[i].size());
    for(int j = 0; j < num_canvases; ++j)
    {
      m_canvases[i][j]->SetBackgroundColor(vtkm_color);
    }
  }
}

void 
vtkhRenderer::Composite(const int &num_images)
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());

  m_compositor->SetCompositeMode(Compositor::Z_BUFFER_SURFACE);

  for(int i = 0; i < num_images; ++i)
  {
    for(int dom = 0; dom < num_domains; ++dom)
    {
      float* color_buffer = &GetVTKMPointer(m_canvases[dom][i]->GetColorBuffer())[0][0]; 
      float* depth_buffer = GetVTKMPointer(m_canvases[dom][i]->GetDepthBuffer()); 
      int height = m_canvases[dom][i]->GetHeight();
      int width = m_canvases[dom][i]->GetWidth();

      m_compositor->AddImage(color_buffer,
                             depth_buffer,
                             width,
                             height);
    } //for dom

    Image result = m_compositor->Composite();
    const std::string image_name = "output.png";
#ifdef PARALLEL
    if(VTKH::GetMPIRank() == 0)
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
vtkhRenderer::Render()
{
  if(m_mapper.get() == 0)
  {
    std::string msg = "Renderer Error: no renderer was set by sub-class"; 
    throw Error(msg);
  }
  

  // check to see if any cameras were set.
  // if not, then just render with the default camera.
  bool using_default_camera = false;
  if(m_cameras.size() == 0)
  {
    m_cameras.push_back(m_default_camera);
    using_default_camera = true;
  }
  
  this->SetupCanvases();

  int total_images = static_cast<int>(m_cameras.size());
  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  int images_processed = 0;
  
  while(images_processed < total_images)
  {
    int images_remaining = total_images - images_processed;
    int current_batch_size = std::min(images_remaining, m_batch_size);   
    for(int dom = 0; dom < num_domains; ++dom)
    {
      vtkm::cont::DataSet data_set; 
      int domain_id;
      m_input->GetDomain(dom, data_set, domain_id);
      const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
      const vtkm::cont::Field &field = data_set.GetField(m_field_index);
      const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();
      for(int i = 0; i < current_batch_size; ++i)
      {
        // paint
        vtkmCanvasPtr p_canvas = m_canvases[dom][i];
        int current_image = images_processed + i;
        vtkmCamera camera = m_cameras[i]; 
        m_mapper->SetCanvas(&(*p_canvas));
        m_mapper->RenderCells(cellset,
                              coords,
                              field,
                              m_color_table,
                              camera,
                              m_range);
      }
    }

    this->Composite(current_batch_size);
    //
    // TODO: output the images to png or into the data set
    //
    images_processed += current_batch_size;
  }

  // remove the default camera
  if(using_default_camera)
  {
    m_cameras.clear();
  }

}
 
void 
vtkhRenderer::PreExecute() 
{
  m_bounds = this->m_input->GetGlobalBounds();
  m_default_camera.ResetToBounds(m_bounds);
  
  // Look for a provided field 
  if(m_field_name != "")
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet data_set;
    m_input->GetDomain(0, data_set, domain_id);
    m_field_index = data_set.GetFieldIndex(m_field_name);
  }

  vtkm::cont::ArrayHandle<vtkm::Range> ranges = m_input->GetGlobalRange(m_field_index);
  int num_components = ranges.GetPortalControl().GetNumberOfValues();
  //
  // current vtkm renderers only supports single component scalar fields
  //
  assert(num_components == 1);
  m_range = ranges.GetPortalControl().Get(0);

  m_mapper->SetActiveColorTable(m_color_table);
}

void 
vtkhRenderer::PostExecute() 
{
}

void 
vtkhRenderer::DoExecute() 
{
  Render();
}

} // namespace vtkh
