#include "ANARIVolumeRenderer.hpp"

#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/compositing/Compositor.hpp>
#include <vtkh/Logger.hpp>


#include <memory>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

#include <vtkm/interop/anari/ANARIMapperTriangles.h>
#include <vtkm/interop/anari/ANARIMapperGlyphs.h>
#include <vtkm/interop/anari/ANARIMapperVolume.h>
#include <vtkm/interop/anari/ANARIScene.h>

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/ConnectivityProxy.h>
#include <vtkh/compositing/PartialCompositor.hpp>
#include <vtkm/rendering/raytracing/VolumeRendererStructured.h>
#include <vtkm/rendering/raytracing/RayOperations.h>
#include <vtkm/rendering/raytracing/Camera.h>
#include <vtkh/compositing/VolumePartial.hpp>

#include <png_utils/ascent_png_encoder.hpp>

#define VTKH_OPACITY_CORRECTION 10.f

namespace vtkh {

namespace detail
{

static void StatusFunc(const void* userData,
                       ANARIDevice /*device*/,
                       ANARIObject source,
                       ANARIDataType /*sourceType*/,
                       ANARIStatusSeverity severity,
                       ANARIStatusCode /*code*/,
                       const char* message)
{
  bool verbose = *(bool*)userData;
  if (!verbose)
    return;

  if (severity == ANARI_SEVERITY_FATAL_ERROR)
  {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
  }
  else if (severity == ANARI_SEVERITY_ERROR)
  {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  }
  else if (severity == ANARI_SEVERITY_WARNING)
  {
    fprintf(stderr, "[WARN ][%p] %s\n", source, message);
  }
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
  {
    fprintf(stderr, "[PERF ][%p] %s\n", source, message);
  }
  else if (severity == ANARI_SEVERITY_INFO)
  {
    fprintf(stderr, "[INFO ][%p] %s\n", source, message);
  }
  else if (severity == ANARI_SEVERITY_DEBUG)
  {
    fprintf(stderr, "[DEBUG][%p] %s\n", source, message);
  }
}

static anari_cpp::Device 
anari_device_load()
{
  static char* libraryName 
    = std::getenv("ANARI_LIBRARY")           ? std::getenv("ANARI_LIBRARY") 
    : std::getenv("VTKM_ANARI_LIBRARY")      ? std::getenv("VTKM_ANARI_LIBRARY") 
    : std::getenv("VTKM_TEST_ANARI_LIBRARY") ? std::getenv("VTKM_TEST_ANARI_LIBRARY") // fall back to the old environment variable
    : nullptr;
  std::cout << "Library loaded: " << (libraryName ? libraryName : "helide") << std::endl;
  static bool verbose = std::getenv("VTKM_ANARI_VERBOSE") != nullptr;
  static bool debug = std::getenv("VTKM_ANARI_DEBUG_DEVICE") != nullptr;
  static char* trace_dir = std::getenv("VTKM_ANARI_DEBUG_TRACE_DIR");

  auto lib = anari_cpp::loadLibrary(libraryName ? libraryName : "helide", StatusFunc, &verbose);
  auto dev = anari_cpp::newDevice(lib, "default");
  anari_cpp::unloadLibrary(lib);

  return dev;
}

void
set_tfn(vtkm::interop::anari::ANARIMapper& mapper, 
        anari_cpp::Device& device,
        vtkm::cont::ColorTable& color_table,
        vtkm::Range& scalar_range)
{
  //auto colorArray = anari_cpp::newArray1D(device, ANARI_FLOAT32_VEC3, 3);
  //auto* colors = anari_cpp::map<vtkm::Vec3f_32>(device, colorArray);
  //colors[0] = vtkm::Vec3f_32(0.f, 0.f, 1.f);
  //colors[1] = vtkm::Vec3f_32(0.f, 1.f, 0.f);
  //colors[2] = vtkm::Vec3f_32(1.f, 0.f, 0.f);
  //anari_cpp::unmap(device, colorArray);

  //auto opacityArray = anari_cpp::newArray1D(device, ANARI_FLOAT32, 2);
  //auto* opacities = anari_cpp::map<float>(device, opacityArray);
  //opacities[0] = 0.f;
  //opacities[1] = 1.f;
  //anari_cpp::unmap(device, opacityArray);

  //mapper.SetColorTable(color_table);
  //mapper.SetANARIColorMap(colorArray, opacityArray, true);
  //mapper.SetANARIColorMapValueRange(vtkm::Vec2f_32(0.f, 10.f));
  //mapper.SetANARIColorMapOpacityScale(0.5f);

  constexpr int resolution = 256;

  constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);
  vtkm::cont::ArrayHandle<vtkm::Vec4ui_8> temp;
  {
    color_table.Sample(resolution, temp);
  }
  auto colorPortal = temp.ReadPortal();

  // Create the color and opacity arrays
  auto colorArray   = anari_cpp::newArray1D(device, ANARI_FLOAT32_VEC3, resolution);
  auto* colors      = anari_cpp::map<vtkm::Vec3f_32>(device, colorArray  );
  auto opacityArray = anari_cpp::newArray1D(device, ANARI_FLOAT32,      resolution);
  auto* opacities   = anari_cpp::map<vtkm::Float32 >(device, opacityArray);
  for (vtkm::Id i = 0; i < resolution; ++i)
  {
    auto color = colorPortal.Get(i);
    colors[i] = vtkm::Vec3f_32(color[0], color[1], color[2]) * conversionToFloatSpace;
    opacities[i] = color[3] * conversionToFloatSpace;
  }

  anari_cpp::unmap(device, colorArray);
  anari_cpp::unmap(device, opacityArray);

  mapper.SetANARIColorMap(colorArray, opacityArray, true);
  if (scalar_range.IsNonEmpty()) {
    mapper.SetANARIColorMapValueRange(vtkm::Vec2f_32(scalar_range.Min, scalar_range.Max));
  }
  else {
    auto range = color_table.GetRange();
    mapper.SetANARIColorMapValueRange(vtkm::Vec2f_32(range.Min, range.Max));
  }
  mapper.SetANARIColorMapOpacityScale(1.0f);
}

struct VisOrdering
{
  int m_rank;
  int m_domain_index;
  int m_order;
  float m_minz;
};

struct DepthOrder
{
  inline bool operator()(const VisOrdering &lhs, const VisOrdering &rhs)
  {
    return lhs.m_minz < rhs.m_minz;
  }
};

struct RankOrder
{
  inline bool operator()(const VisOrdering &lhs, const VisOrdering &rhs)
  {
    if(lhs.m_rank < rhs.m_rank)
    {
      return true;
    }
    else if(lhs.m_rank == rhs.m_rank)
    {
      return lhs.m_domain_index < rhs.m_domain_index;
    }
    return false;
  }
};

} //  namespace detail

ANARIVolumeRenderer::ANARIVolumeRenderer()
{
  m_device = detail::anari_device_load();
  m_renderer = anari_cpp::newObject<anari_cpp::Renderer>(m_device,"default");
  m_frame = anari_cpp::newObject<anari_cpp::Frame>(m_device);

  for (auto& light : m_lights)
  {
    anari_cpp::release(m_device, light);
  }
  m_lights.clear();

  // create default m_lights
  anari_cpp::Light sun = anari_cpp::newObject<anari_cpp::Light>(m_device, "directional");
  anari_cpp::setParameter(m_device, sun, "direction", vtkm::Vec3f_32(0.0f, -1.0f, 0.0f));
  anari_cpp::setParameter(m_device, sun, "irradiance", 2.f);
  anari_cpp::setParameter(m_device, sun, "angularDiameter", 0.00925f);
  anari_cpp::setParameter(m_device, sun, "radiance", 1.f);
  anari_cpp::commitParameters(m_device, sun);
  m_lights.push_back(sun);
  //old:
  typedef vtkm::rendering::MapperVolume TracerType;
  m_tracer = std::make_shared<TracerType>();
  this->m_mapper = m_tracer;
  //m_tracer->SetCompositeBackground(false);
  ////
  //// add some default opacity to the color table
  ////
  //m_color_table.AddPointAlpha(0.0f, .02);
  //m_color_table.AddPointAlpha(.0f, .5);
  m_num_samples = 100.f;
  //m_has_unstructured = false;
}

ANARIVolumeRenderer::~ANARIVolumeRenderer()
{
}

void
ANARIVolumeRenderer::Update()
{
  VTKH_DATA_OPEN(this->GetName());
#ifdef VTKH_ENABLE_LOGGING
  VTKH_DATA_ADD("device", GetCurrentDevice());
  long long int in_cells = this->m_input->GetNumberOfCells();
  VTKH_DATA_ADD("input_cells", in_cells);
  VTKH_DATA_ADD("input_domains", this->m_input->GetNumberOfDomains());
  int in_topo_dims;
  //old:
  //bool in_structured = this->m_input->IsStructured(in_topo_dims);
  //if(in_structured)
  //{
  //  VTKH_DATA_ADD("in_topology", "structured");
  //}
  //else
  //{
  //  VTKH_DATA_ADD("in_topology", "unstructured");
  //}
#endif

  PreExecute();
  DoExecute();
  PostExecute();

  VTKH_DATA_CLOSE();
}

void ANARIVolumeRenderer::SetColorTable(const vtkm::cont::ColorTable &color_table)
{
  m_color_table = color_table;
}

void
ANARIVolumeRenderer::DoExecute()
{
  //old from Qi:
  //shouldn't need this since calling 
  //Renderer::PreExecute sets m_range & m_bounds to global values
  // Compute value range if necessary
  //if (!scalar_range.IsNonEmpty()) 
  //{
  //  auto ranges = dset.GetGlobalRange(field_name);
  //  auto size = ranges.GetNumberOfValues();
  //  if (size != 1) 
  //  {
  //    ASCENT_ERROR("Anari Volume only supports scalar fields");
  //  }
  //  auto portal = ranges.ReadPortal();
  //  for (int cc = 0; cc < size; ++cc)
  //  {
  //    auto range = portal.Get(cc);
  //    scalar_range.Include(range);
  //    break;
  //  }
  //}

  // Build Scene
  vtkm::interop::anari::ANARIScene scene(m_device);
  for (int i = 0; i < m_input->GetNumberOfDomains(); ++i)
  {
    //std::cerr << "CELLSET: " << std::endl;
    //m_input->GetDomain(i).GetCellSet().PrintSummary(std::cerr);
    //std::cerr << "COORDSET: " << std::endl;
    //m_input->GetDomain(i).GetCoordinateSystem().PrintSummary(std::cerr);
    //std::cerr << "FIELD: " << std::endl;
    //m_input->GetDomain(i).GetField(m_field_name).PrintSummary(std::cerr);
    if(m_input->IsUnstructured())
    {
      std::cerr << "in unstructured" << std::endl;
      auto& mIso = scene.AddMapper(vtkm::interop::anari::ANARIMapperTriangles(m_device));
      mIso.SetName(("isosurface_" + std::to_string(i)).c_str());
      mIso.SetActor({ 
        m_input->GetDomain(i).GetCellSet(), 
        m_input->GetDomain(i).GetCoordinateSystem(), 
        m_input->GetDomain(i).GetField(m_field_name) 
      });
      mIso.SetCalculateNormals(true);
      mIso.SetColorTable(m_color_table);
      //detail::set_tfn(mIso,m_device,m_color_table,m_range);
    }

    auto& mVol = scene.AddMapper(vtkm::interop::anari::ANARIMapperVolume(m_device));
    mVol.SetName(("volume_" + std::to_string(i)).c_str());
    mVol.SetActor({ 
      m_input->GetDomain(i).GetCellSet(), 
      m_input->GetDomain(i).GetCoordinateSystem(), 
      m_input->GetDomain(i).GetField(m_field_name) 
    });
    mVol.SetColorTable(m_color_table);
    //detail::set_tfn(mVol,m_device,m_color_table,m_range);
  }
  int num_renders = static_cast<int>(m_renders.size());
  for(int i = 0; i < num_renders; ++i)
  {
    vtkm::rendering::Camera cam = m_renders[i].GetCamera();
    vtkm::rendering::Canvas &canvas = m_renders[i].GetCanvas();
    vtkm::Vec4f_32 background = m_renders[i].GetBackgroundColor().Components;
    std::string img_name = m_renders[i].GetImageName();
    vtkm::Float32 height = m_renders[i].GetHeight();
    vtkm::Float32 width = m_renders[i].GetWidth();
    // Finalize
    //render(scene);
    // m_renderer parameters
    anari_cpp::setParameter(m_device, m_renderer, "background", background);
    anari_cpp::setParameter(m_device, m_renderer, "pixelSamples", m_num_samples);
    anari_cpp::setParameter(m_device, m_renderer, "ambientRadiance", 0.8f);
    anari_cpp::commitParameters(m_device, m_renderer);
  
    // TODO support all camera parameters
    //    -- missing parameters: xpan, ypan (through imageRegion)
    //
    const auto cam_zoom = cam.GetZoom();
    const auto cam_type = cam.GetMode() == vtkm::rendering::Camera::Mode::ThreeD ? "perspective" : "orthographic";
    const auto cam_dir = cam.GetLookAt() - cam.GetPosition();
    // TODO: what is the correct way to apply zoom?
    const auto cam_pos = cam_zoom > 0
      ? cam.GetLookAt() - cam_dir / cam_zoom
      : cam.GetPosition();
    const auto cam_up  = cam.GetViewUp();
    const auto cam_range = cam.GetClippingRange();
    anari_cpp::Camera camera = anari_cpp::newObject<anari_cpp::Camera>(m_device, cam_type);
    anari_cpp::setParameter(m_device, camera, "aspect",  float(width) / float(height));
    anari_cpp::setParameter(m_device, camera, "position",  cam_pos);
    anari_cpp::setParameter(m_device, camera, "direction", cam_dir);
    anari_cpp::setParameter(m_device, camera, "up", cam_up);
    anari_cpp::setParameter(m_device, camera, "near", cam_range.Min);
    anari_cpp::setParameter(m_device, camera, "far",  cam_range.Max);
    if (cam_type == "perspective")
    {
      anari_cpp::setParameter(m_device, camera, "fov", cam.GetFieldOfView() / 180.0 * vtkm::Pi());
    }
    else
    {
      anari_cpp::setParameter(m_device, camera, "height", cam.GetXScale() / width * height);
    }
    anari_cpp::commitParameters(m_device, camera);

    // commit world with lights
    auto world = scene.GetANARIWorld();
    anari_cpp::setAndReleaseParameter(m_device, world, "light", 
    anari_cpp::newArray1D(m_device, m_lights.data(), m_lights.size()));
    anari_cpp::commitParameters(m_device, world);

    // m_frame parameters
    vtkm::Vec2ui_32 img_size = vtkm::Vec2ui_32(width,height);
    anari_cpp::setParameter(m_device, m_frame, "size", img_size);
    //anari_cpp::setParameter(m_device, m_frame, "channel.color", ANARI_UFIXED8_VEC4);
    anari_cpp::setParameter(m_device, m_frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari_cpp::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
    anari_cpp::setParameter(m_device, m_frame, "world", world);
    anari_cpp::setParameter(m_device, m_frame, "camera", camera);
    anari_cpp::setParameter(m_device, m_frame, "renderer", m_renderer);
    anari_cpp::commitParameters(m_device, m_frame);

    // render and wait for completion
    anari_cpp::render(m_device, m_frame);
    anari_cpp::wait(m_device, m_frame);

    //const auto a_colors = anari_cpp::map<uint32_t>(m_device, m_frame, "channel.color");
    const auto a_colors = anari_cpp::map<vtkm::Vec4f_32>(m_device, m_frame, "channel.color");
    const auto a_depths = anari_cpp::map<vtkm::Float32>(m_device, m_frame, "channel.depth");

    //ascent::PNGEncoder encoder;
    //encoder.Encode((float *)a_colors.data, a_colors.width, a_colors.height);
    //encoder.Save("encoder_image.png");
    auto v_colors = canvas.GetColorBuffer().WritePortal();
    auto v_depths = canvas.GetDepthBuffer().WritePortal();
    const float *d_pixels = anari::map<float>(m_device, m_frame, "channel.depth").data;
    int size = width*height;
    for(int pixel = 0; pixel < size; ++pixel)
    {
      int color_index = pixel*4;
      //std::cerr << "color index: " << color_index << std::endl;
      vtkm::Vec4f_32 color;
      //color[0] = a_colors.data[color_index];
      //color[1] = a_colors.data[color_index+1];
      //color[2] = a_colors.data[color_index+2];
      //color[3] = a_colors.data[color_index+3];
      v_colors.Set(pixel,a_colors.data[pixel]);
      //vtkm::rendering::Color color;
      //color.SetComponentFromByte(0, a_colors.data[color_index]);
      //color.SetComponentFromByte(1, a_colors.data[color_index + 1]);
      //color.SetComponentFromByte(2, a_colors.data[color_index + 2]);
      //color.SetComponentFromByte(3, a_colors.data[color_index + 3]);
      //std::cerr << "get depth" << std::endl;
      vtkm::Float32 d = d_pixels[pixel];
      //if(d < 10000)
      //  std::cerr << "set depth: " << d << " at pixel: " << pixel << std::endl;
      v_depths.Set(pixel,d);

    }
     
    m_mapper->SetCanvas(&canvas);
    anari_cpp::unmap(m_device, m_frame, "channel.color");
    anari_cpp::unmap(m_device, m_frame, "channel.depth");

    // release resources
    anari_cpp::release(m_device, camera);
  } 
  if(m_do_composite)
  {
    this->Composite(num_renders);
  }
}


void
ANARIVolumeRenderer::PreExecute()
{
  Renderer::PreExecute();

  vtkm::Vec<vtkm::Float32,3> extent;
  extent[0] = static_cast<vtkm::Float32>(this->m_bounds.X.Length());
  extent[1] = static_cast<vtkm::Float32>(this->m_bounds.Y.Length());
  extent[2] = static_cast<vtkm::Float32>(this->m_bounds.Z.Length());
  vtkm::Float32 dist = vtkm::Magnitude(extent) / m_num_samples;
  m_sample_dist = dist;
}

void
ANARIVolumeRenderer::PostExecute()
{
  // do nothing and override compositing since
  // we already did it
}

float
ANARIVolumeRenderer::FindMinDepth(const vtkm::rendering::Camera &camera,
                                 const vtkm::Bounds &bounds) const
{

  vtkm::Vec<vtkm::Float64,3> center = bounds.Center();
  vtkm::Vec<vtkm::Float64,3> fcenter;
  fcenter[0] = static_cast<vtkm::Float32>(center[0]);
  fcenter[1] = static_cast<vtkm::Float32>(center[1]);
  fcenter[2] = static_cast<vtkm::Float32>(center[2]);
  vtkm::Vec<vtkm::Float32,3> pos = camera.GetPosition();
  vtkm::Float32 dist = vtkm::Magnitude(fcenter - pos);
  return dist;
}

void
ANARIVolumeRenderer::Composite(const int &num_images)
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());

  m_compositor->SetCompositeMode(Compositor::VIS_ORDER_BLEND);

  FindVisibilityOrdering();

  for(int i = 0; i < num_images; ++i)
  {
    float* color_buffer =
      &GetVTKMPointer(m_renders[i].GetCanvas().GetColorBuffer())[0][0];
    float* depth_buffer =
      GetVTKMPointer(m_renders[i].GetCanvas().GetDepthBuffer());
    int height = m_renders[i].GetCanvas().GetHeight();
    int width = m_renders[i].GetCanvas().GetWidth();

    m_compositor->AddImage(color_buffer,
                           depth_buffer,
                           width,
                           height,
                           m_visibility_orders[i][0]);

    Image result = m_compositor->Composite();
    const std::string image_name = m_renders[i].GetImageName() + ".png";
#ifdef VTKH_PARALLEL
    if(vtkh::GetMPIRank() == 0)
    {
#endif
      ImageToCanvas(result, m_renders[i].GetCanvas(), true);
#ifdef VTKH_PARALLEL
    }
#endif
    m_compositor->ClearImages();
  } // for image
}

void
ANARIVolumeRenderer::DepthSort(int num_domains,
                          std::vector<float> &min_depths,
                          std::vector<int> &local_vis_order)
{
  if(min_depths.size() != num_domains)
  {
    throw Error("min depths size does not equal the number of domains");
  }
  if(local_vis_order.size() != num_domains)
  {
    throw Error("local vis order not equal to number of domains");
  }
#ifdef VTKH_PARALLEL
  int root = 0;
  MPI_Comm comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  int num_ranks = vtkh::GetMPISize();
  int rank = vtkh::GetMPIRank();
  int *domain_counts = NULL;
  int *domain_offsets = NULL;
  int *vis_order = NULL;
  float *depths = NULL;

  if(rank == root)
  {
    domain_counts = new int[num_ranks];
    domain_offsets = new int[num_ranks];
  }

  MPI_Gather(&num_domains,
             1,
             MPI_INT,
             domain_counts,
             1,
             MPI_INT,
            root,
             comm);

  int depths_size = 0;
  if(rank == root)
  {
    //scan for dispacements
    domain_offsets[0] = 0;
    for(int i = 1; i < num_ranks; ++i)
    {
      domain_offsets[i] = domain_offsets[i - 1] + domain_counts[i - 1];
    }

    for(int i = 0; i < num_ranks; ++i)
    {
      depths_size += domain_counts[i];
    }

    depths = new float[depths_size];

  }

  MPI_Gatherv(&min_depths[0],
              num_domains,
              MPI_FLOAT,
              depths,
              domain_counts,
              domain_offsets,
              MPI_FLOAT,
              root,
              comm);

  if(rank == root)
  {
    std::vector<detail::VisOrdering> order;
    order.resize(depths_size);

    for(int i = 0; i < num_ranks; ++i)
    {
      for(int c = 0; c < domain_counts[i]; ++c)
      {
        int index = domain_offsets[i] + c;
        order[index].m_rank = i;
        order[index].m_domain_index = c;
        order[index].m_minz = depths[index];
      }
    }

    std::sort(order.begin(), order.end(), detail::DepthOrder());

    for(int i = 0; i < depths_size; ++i)
    {
      order[i].m_order = i;
    }

    std::sort(order.begin(), order.end(), detail::RankOrder());

    vis_order = new int[depths_size];
    for(int i = 0; i < depths_size; ++i)
    {
      vis_order[i] = order[i].m_order;
    }
  }

  MPI_Scatterv(vis_order,
               domain_counts,
               domain_offsets,
               MPI_INT,
               &local_vis_order[0],
               num_domains,
               MPI_INT,
               root,
               comm);

  if(rank == root)
  {
    delete[] domain_counts;
    delete[] domain_offsets;
    delete[] vis_order;
    delete[] depths;
  }
#else

  std::vector<detail::VisOrdering> order;
  order.resize(num_domains);

  for(int i = 0; i < num_domains; ++i)
  {
      order[i].m_rank = 0;
      order[i].m_domain_index = i;
      order[i].m_minz = min_depths[i];
  }
  std::sort(order.begin(), order.end(), detail::DepthOrder());

  for(int i = 0; i < num_domains; ++i)
  {
    order[i].m_order = i;
  }

  std::sort(order.begin(), order.end(), detail::RankOrder());

  for(int i = 0; i < num_domains; ++i)
  {
    local_vis_order[i] = order[i].m_order;
  }
#endif
}

void
ANARIVolumeRenderer::FindVisibilityOrdering()
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  const int num_cameras = static_cast<int>(m_renders.size());
  m_visibility_orders.resize(num_cameras);

  for(int i = 0; i < num_cameras; ++i)
  {
    m_visibility_orders[i].resize(num_domains);
  }

  //
  // In order for parallel volume rendering to composite correctly,
  // we nee to establish a visibility ordering to pass to IceT.
  // We will transform the data extents into camera space and
  // take the minimum z value. Then sort them while keeping
  // track of rank, then pass the list in.
  //
  std::vector<float> min_depths;
  min_depths.resize(num_domains);

  for(int i = 0; i < num_cameras; ++i)
  {
    const vtkm::rendering::Camera &camera = m_renders[i].GetCamera();
    for(int dom = 0; dom < num_domains; ++dom)
    {
      vtkm::Bounds bounds = this->m_input->GetDomainBounds(dom);
      min_depths[dom] = FindMinDepth(camera, bounds);
    }

    DepthSort(num_domains, min_depths, m_visibility_orders[i]);

  } // for each camera
}

void
ANARIVolumeRenderer::SetNumberOfSamples(const int num_samples)
{
  if(num_samples < 1)
  {
    throw Error("Volume rendering samples must be greater than 0");
  }
  m_num_samples = num_samples;
}

Renderer::vtkmCanvasPtr
ANARIVolumeRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

std::string
ANARIVolumeRenderer::GetName() const
{
  return "vtkh::ANARIVolumeRenderer";
}

} // namespace vtkh
