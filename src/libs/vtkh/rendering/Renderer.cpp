#include "Renderer.hpp"
#include <vtkh/compositing/Compositor.hpp>

#include <vtkh/Logger.hpp>
#include <vtkh/utils/viskores_array_utils.hpp>
#include <vtkh/utils/viskores_dataset_info.hpp>
#include <viskores/rendering/raytracing/Logger.h>
#include <viskores/rendering/MapperCylinder.h>
#include <viskores/rendering/MapperPoint.h>
#include <viskores/rendering/MapperWireframer.h>


#include <png_utils/ascent_png_encoder.hpp>

namespace vtkh {

Renderer::Renderer()
  : m_do_composite(true),
    m_color_table("Cool to Warm"),
    m_field_index(0),
    m_has_color_table(true),
    m_is_discrete(false)
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
Renderer::SetDiscrete()
{
  m_is_discrete = true;
}

bool
Renderer::IsDiscrete() const
{
  return m_is_discrete;
}

bool
Renderer::IsMeshRenderer() const
{
  bool is_mesh = false;

  if(std::dynamic_pointer_cast<viskores::rendering::MapperWireframer>(m_mapper) != nullptr)
  {
    is_mesh = true;
  }
  return is_mesh;
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

void Renderer::SetColorTable(const viskores::cont::ColorTable &color_table)
{
  m_color_table = color_table;
}

viskores::cont::ColorTable Renderer::GetColorTable() const
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
    float* color_buffer = &GetVISKORESPointer(m_renders[i].GetCanvas().GetColorBuffer())[0][0];
    float* depth_buffer = GetVISKORESPointer(m_renders[i].GetCanvas().GetDepthBuffer());

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
    viskores::cont::ArrayHandle<viskores::Range> ranges = m_input->GetGlobalRange(m_field_name);
    int num_components = ranges.GetNumberOfValues();
    //
    // current viskores renderers only supports single component scalar fields
    //
    if(num_components != 1)
    {
      std::stringstream msg;
      msg<<"Renderer '"<<this->GetName()<<"' cannot render a field with ";
      msg<<"'"<<num_components<<"' components. Field must be a scalar field.";
      throw Error(msg.str());
    }

    viskores::Range global_range = ranges.ReadPortal().Get(0);
    // a min or max may be been set by the user, check to see
    if(m_range.Min == viskores::Infinity64())
    {
      m_range.Min = global_range.Min;
    }

    if(m_range.Max == viskores::NegativeInfinity64())
    {
      m_range.Max = global_range.Max;
    }
  }

  m_bounds = m_input->GetGlobalBounds();
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

  bool is_lines = m_input->IsLineMesh();
  //TODO: 
  //deal with 1D lines when viskores updated: https://github.com/Viskores/viskores/issues/164
  if(is_lines && !IsMeshRenderer())
  { 
    typedef viskores::rendering::MapperCylinder TracerType;
    auto mapper = std::make_shared<TracerType>();
    viskores::Bounds bounds = m_input->GetBounds();
    viskores::FloatDefault diagonal = viskores::Magnitude(bounds.MaxCorner() - bounds.MinCorner());
    //TODO: user input radius?
    mapper->SetRadius(0.001 * diagonal);
    this->m_mapper = mapper;
  }

  int total_renders = static_cast<int>(m_renders.size());
  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  for(int dom = 0; dom < num_domains; ++dom)
  {
    viskores::cont::DataSet data_set;
    viskores::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);
    if(!data_set.HasField(m_field_name))
    {
      continue;
    }

    const viskores::cont::UnknownCellSet &cellset = data_set.GetCellSet();
    const viskores::cont::Field &field = data_set.GetField(m_field_name);
    const viskores::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();

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

      Render::viskoresCanvas &canvas = m_renders[i].GetCanvas();
      const viskoresCamera &camera = m_renders[i].GetCamera();
      bool tile_image = false;
      if (m_renders[i].GetTileImage())
      {
        if (canvas.GetWidth() > m_renders[i].GetTileWidth() ||
            canvas.GetHeight() > m_renders[i].GetTileWidth())
        {
          tile_image = true;
        }
      }
      if (tile_image)
      {
        std::cerr << "Calling RenderTiled" << std::endl;
        RenderTiled(canvas,
                    camera,
                    cellset,
                    field,
                    coords,
                    data_set,
		    m_renders[i].GetTileWidth());
      }
      else
      {
        std::cerr << "Calling RenderCells" << std::endl;
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


}

void
Renderer::ImageToCanvas(Image &image, viskores::rendering::Canvas &canvas, bool get_depth)
{
  const int width = canvas.GetWidth();
  const int height = canvas.GetHeight();
  const int size = width * height;
  const int color_size = size * 4;
  float* color_buffer = &GetVISKORESPointer(canvas.GetColorBuffer())[0][0];
  float one_over_255 = 1.f / 255.f;
#ifdef VTKH_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(int i = 0; i < color_size; ++i)
  {
    color_buffer[i] = static_cast<float>(image.m_pixels[i]) * one_over_255;
  }

  float* depth_buffer = GetVISKORESPointer(canvas.GetDepthBuffer());
  if(get_depth) memcpy(depth_buffer, &image.m_depths[0], sizeof(float) * size);
}

void
Renderer::RenderTiled(Render::viskoresCanvas &canvas,
                      const viskoresCamera &camera,
                      const viskores::cont::UnknownCellSet &cellset,
                      const viskores::cont::Field &field,
                      const viskores::cont::CoordinateSystem &coords,
                      viskores::cont::DataSet &data_set,
		      const viskores::Int32 tile_width)
{
  // Calculate the tiling parameters.
  const int x_tile_size = tile_width;
  const int y_tile_size = tile_width;
  const int nx_canvas = canvas.GetWidth();
  const int ny_canvas = canvas.GetHeight();
  const int nx_tiles = int(double(nx_canvas - 1) / double(x_tile_size)) + 1;
  const int ny_tiles = int(double(ny_canvas - 1) / double(y_tile_size)) + 1;
  std::cerr << "nx_tiles=" << nx_tiles << ",ny_tiles=" << ny_tiles << std::endl;

  // Create a canvas for doing the tiling.
  Render::viskoresCanvas *tile_canvas = new viskores::rendering::CanvasRayTracer;
  tile_canvas->SetBackgroundColor(canvas.GetBackgroundColor());
  tile_canvas->SetForegroundColor(canvas.GetForegroundColor());
  tile_canvas->ResizeBuffers(x_tile_size, y_tile_size);
  m_mapper->SetCanvas(tile_canvas);

  viskoresCamera tile_camera = camera;
  std::cerr << "tile_camera.GetZoom()=" << tile_camera.GetZoom() << std::endl;
  std::cerr << "tile_camera.GetPan()=" << tile_camera.GetPan()[0] << "," << tile_camera.GetPan()[1] << std::endl;
  viskores::Float64 zoom_user = tile_camera.GetZoom();
  viskores::Float64 xpan_user = tile_camera.GetPan()[0];
  viskores::Float64 ypan_user = tile_camera.GetPan()[1];

  // Calculate the tile zoom factor and the zoom factor for viskores.
  const double tile_zoom = double(ny_canvas) / double(y_tile_size);
  viskores::Float64 zoom = log(tile_camera.GetZoom() * tile_zoom) / log(4.);
  std::cerr << "tile_zoom=" << tile_zoom << ",zoom=" << zoom << std::endl;

  // Calculate the fraction of the last tile that is each direction.
  const double nx_extra = double((nx_tiles * x_tile_size) - nx_canvas) / double(x_tile_size);
  const double ny_extra= double((ny_tiles * y_tile_size) - ny_canvas) / double(y_tile_size);

  const double xpan_init  = xpan_user * (double(nx_canvas) / double(ny_canvas)) * (double(y_tile_size) / double(x_tile_size)) + double(nx_tiles - 1 - nx_extra) / (double(tile_zoom) * tile_camera.GetZoom());
  const double ypan_init  = ypan_user + double(ny_tiles - 1 - ny_extra) / (double(tile_zoom) * tile_camera.GetZoom());
  const double xpan_delta = 2. / (tile_camera.GetZoom() * double(tile_zoom));
  const double ypan_delta = 2. / (tile_camera.GetZoom() * double(tile_zoom));
  std::cerr << "xpan_init=" << xpan_init << ",ypan_init=" << ypan_init << ",xpan_delta=" << xpan_delta << ",ypan_delta=" << ypan_delta << std::endl;

  float* color_buffer = &GetVISKORESPointer(canvas.GetColorBuffer())[0][0];
  float* depth_buffer = GetVISKORESPointer(canvas.GetDepthBuffer());

  int remaining_ny_canvas = ny_canvas;
  viskores::Float64 ypan = ypan_init;
  for(int j = 0; j < ny_tiles; ++j)
  {
    int remaining_nx_canvas = nx_canvas;
    viskores::Float64 xpan = xpan_init;
    for(int i = 0; i < nx_tiles; ++i)
    {
      // Pan and Zoom in viskores are relative to the current values.
      // These 2 command set zoom = 1. and pan = (0., 0.).
      tile_camera.Zoom(log(1. / tile_camera.GetZoom()) / log(4.));
      tile_camera.Pan(-tile_camera.GetPan()[0], -tile_camera.GetPan()[1]);

      // Now we have the pan and zoom set to the default, we can set it.
      tile_camera.Zoom(zoom);
      tile_camera.Pan(xpan, ypan);
      std::cerr << "2: camera zoom=" << tile_camera.GetZoom() << std::endl;
      std::cerr << "2: camera pan=" << tile_camera.GetPan()[0] << "," << tile_camera.GetPan()[1] << std::endl;

      // Render the tile.
      tile_canvas->Clear();
      m_mapper->RenderCells(cellset,
                            coords,
                            field,
                            m_color_table,
                            tile_camera,
                            m_range);

      // Copy the image from the tile into the output buffer. Note that
      // the last tile in each row and all the tiles in the last row may
      // be larger than necessary, so we only copy part we need.
      const float* tile_color_buffer = &GetVISKORESPointer(tile_canvas->GetColorBuffer())[0][0];
      const float* tile_depth_buffer = GetVISKORESPointer(tile_canvas->GetDepthBuffer());
      const int x_max = std::min(x_tile_size, remaining_nx_canvas);
      const int y_max = std::min(y_tile_size, remaining_ny_canvas);
      for(int jj = 0; jj < y_max; ++jj)
      {
        int ll  = jj * x_tile_size * 4;
        int ll2 = jj * x_tile_size;
        int kk  = ((j * y_tile_size + jj) * nx_canvas + i * x_tile_size) * 4;
        int kk2 = (j * y_tile_size + jj) * nx_canvas + i * x_tile_size;

        for(int ii = 0; ii < x_max; ++ii)
        {
          color_buffer[kk]   = tile_color_buffer[ll];
          color_buffer[kk+1] = tile_color_buffer[ll+1];
          color_buffer[kk+2] = tile_color_buffer[ll+2];
          color_buffer[kk+3] = tile_color_buffer[ll+3];
          depth_buffer[kk2] = tile_depth_buffer[ll2];
          kk  += 4;
          kk2 += 1;
          ll  += 4;
          ll2 += 1;
        }
      }
      xpan -= xpan_delta;
      remaining_nx_canvas -= x_tile_size;
    }
    ypan -= ypan_delta;
    remaining_ny_canvas -= y_tile_size;
  }
}

std::vector<Render>
Renderer::GetRenders() const
{
  return m_renders;
}

vtkh::DataSet *
Renderer::GetInput()
{
  return m_input;
}

viskores::Range
Renderer::GetRange() const
{
  return m_range;
}

void
Renderer::SetRange(const viskores::Range &range)
{
  m_range = range;
}

} // namespace vtkh
