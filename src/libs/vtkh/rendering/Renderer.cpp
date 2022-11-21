#include "Renderer.hpp"
#include <vtkh/compositing/Compositor.hpp>

#include <vtkh/Logger.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>
#include <vtkm/rendering/raytracing/Logger.h>

namespace vtkh {

Renderer::Renderer()
  : m_do_composite(true),
    m_color_table("Cool to Warm"),
    m_field_index(0),
    m_has_color_table(true)
{
  m_compositor  = new Compositor();
}

Renderer::~Renderer()
{
  delete m_compositor;
}

void
Renderer::SetShadingOn(bool on)

{
  // do nothing by default;
}

void Renderer::DisableColorBar()
{
  // not all plots have color bars, so
  // we only give the option to turn it off
  m_has_color_table = false;
}

void
Renderer::SetField(const std::string field_name)
{
  m_field_name = field_name;
}

std::string
Renderer::GetFieldName() const
{
  return m_field_name;
}

bool
Renderer::GetHasColorTable() const
{
  return m_has_color_table;
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

void
Renderer::SetRenders(const std::vector<vtkh::Render> &renders)
{
  m_renders = renders;
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

void Renderer::SetColorTable(const vtkm::cont::ColorTable &color_table)
{
  m_color_table = color_table;
}

vtkm::cont::ColorTable Renderer::GetColorTable() const
{
  return m_color_table;
}

void
Renderer::Composite(const int &num_images)
{
  VTKH_DATA_OPEN("Composite");
  m_compositor->SetCompositeMode(Compositor::Z_BUFFER_SURFACE);
  for(int i = 0; i < num_images; ++i)
  {
    float* color_buffer = &GetVTKMPointer(m_renders[i].GetCanvas().GetColorBuffer())[0][0];
    float* depth_buffer = GetVTKMPointer(m_renders[i].GetCanvas().GetDepthBuffer());

    int height = m_renders[i].GetCanvas().GetHeight();
    int width = m_renders[i].GetCanvas().GetWidth();

    m_compositor->AddImage(color_buffer,
                           depth_buffer,
                           width,
                           height);

    Image result = m_compositor->Composite();

#ifdef VTKH_PARALLEL
    if(vtkh::GetMPIRank() == 0)
    {
      ImageToCanvas(result, m_renders[i].GetCanvas(), true);
    }
#else
    ImageToCanvas(result, m_renders[i].GetCanvas(), true);
#endif
    m_compositor->ClearImages();
  } // for image
  VTKH_DATA_CLOSE();
}

void
Renderer::PreExecute()
{
  bool range_set = m_range.IsNonEmpty();
  Filter::CheckForRequiredField(m_field_name);

  if(!range_set)
  {
    // we have not been given a range, so ask the data set
    //vtkm::cont::ArrayHandle<vtkm::Range> ranges = m_input->GetGlobalRange(m_field_name);
    vtkm::cont::Field in_field = m_input->GetGlobalField(m_field_name);
    vtkm::cont::ArrayHandle<vtkm::Range> ranges = in_field.GetRange();
    int num_components = ranges.GetNumberOfValues();
    //
    // current vtkm renderers only supports single component scalar fields
    //
    if(num_components != 1)
    {
      std::stringstream msg;
      msg<<"Renderer '"<<this->GetName()<<"' cannot render a field with ";
      msg<<"'"<<num_components<<"' components. Field must be a scalar field.";
      throw Error(msg.str());
    }

    vtkm::Range global_range = ranges.ReadPortal().Get(0);
    // a min or max may be been set by the user, check to see
    if(m_range.Min == vtkm::Infinity64())
    {
      m_range.Min = global_range.Min;
    }

    if(m_range.Max == vtkm::NegativeInfinity64())
    {
      m_range.Max = global_range.Max;
    }
  }

  m_bounds = GetGlobalBounds(m_input);
}

void
Renderer::Update()
{
  VTKH_DATA_OPEN(this->GetName());
#ifdef VTKH_ENABLE_LOGGING
  long long int in_cells = this->m_input->GetNumberOfCells();
  VTKH_DATA_ADD("input_cells", in_cells);
#endif
  PreExecute();
  DoExecute();
  PostExecute();
  VTKH_DATA_CLOSE();
}

void
Renderer::PostExecute()
{
  int total_renders = static_cast<int>(m_renders.size());
  if(m_do_composite)
  {
    this->Composite(total_renders);
  }
}

void
Renderer::DoExecute()
{
  if(m_mapper.get() == 0)
  {
    std::string msg = "Renderer Error: no renderer was set by sub-class";
    throw Error(msg);
  }

  int total_renders = static_cast<int>(m_renders.size());

  int num_domains = static_cast<int>(m_input->GetNumberOfPartitions());
  for(int dom = 0; dom < num_domains; ++dom)
  {
    vtkm::cont::DataSet data_set;
    data_set = m_input->GetPartition(dom);
    if(!data_set.HasField(m_field_name))
    {
      continue;
    }

    const vtkm::cont::UnknownCellSet &cellset = data_set.GetCellSet();
    const vtkm::cont::Field &field = data_set.GetField(m_field_name);
    const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();

    if(cellset.GetNumberOfCells() == 0)
    {
      continue;
    }

    for(int i = 0; i < total_renders; ++i)
    {
      if(m_renders[i].GetShadingOn())
      {
        this->SetShadingOn(true);
      }
      else
      {
        this->SetShadingOn(false);
      }

      m_mapper->SetActiveColorTable(m_color_table);

      Render::vtkmCanvas &canvas = m_renders[i].GetCanvas();
      const vtkmCamera &camera = m_renders[i].GetCamera();
      m_mapper->SetCanvas(&canvas);
      m_mapper->RenderCells(cellset,
                            coords,
                            field,
                            m_color_table,
                            camera,
                            m_range);
    }
  }


}

void
Renderer::ImageToCanvas(Image &image, vtkm::rendering::Canvas &canvas, bool get_depth)
{
  const int width = canvas.GetWidth();
  const int height = canvas.GetHeight();
  const int size = width * height;
  const int color_size = size * 4;
  float* color_buffer = &GetVTKMPointer(canvas.GetColorBuffer())[0][0];
  float one_over_255 = 1.f / 255.f;
#ifdef VTKH_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(int i = 0; i < color_size; ++i)
  {
    color_buffer[i] = static_cast<float>(image.m_pixels[i]) * one_over_255;
  }

  float* depth_buffer = GetVTKMPointer(canvas.GetDepthBuffer());
  if(get_depth) memcpy(depth_buffer, &image.m_depths[0], sizeof(float) * size);
}

std::vector<Render>
Renderer::GetRenders() const
{
  return m_renders;
}

vtkm::cont::PartitionedDataSet *
Renderer::GetInput()
{
  return m_input;
}

vtkm::Range
Renderer::GetRange() const
{
  return m_range;
}

void
Renderer::SetRange(const vtkm::Range &range)
{
  m_range = range;
}

} // namespace vtkh
