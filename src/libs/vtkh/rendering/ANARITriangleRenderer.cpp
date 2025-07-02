#include "ANARITriangleRenderer.hpp"

#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/compositing/Compositor.hpp>
#include <vtkh/Logger.hpp>


#include <memory>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

#include <vtkm/interop/anari/ANARIMapperTriangles.h>
#include <vtkm/interop/anari/ANARIScene.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/ConnectivityProxy.h>
#include <vtkm/rendering/raytracing/Camera.h>

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

static void 
setColorMap(anari_cpp::Device d, vtkm::interop::anari::ANARIMapper& mapper, vtkm::cont::ColorTable &color_table, vtkm::Range &field_range)
{

  auto colorArray = anari_cpp::newArray1D(d, ANARI_FLOAT32_VEC3, 3);
  auto* colors = anari_cpp::map<vtkm::Vec3f_32>(d, colorArray);
  colors[0] = vtkm::Vec3f_32(0.f, 0.f, 1.f);
  colors[1] = vtkm::Vec3f_32(0.f, 1.f, 0.f);
  colors[2] = vtkm::Vec3f_32(1.f, 0.f, 0.f);
  anari_cpp::unmap(d, colorArray);

  auto opacityArray = anari_cpp::newArray1D(d, ANARI_FLOAT32, 2);
  auto* opacities = anari_cpp::map<float>(d, opacityArray);
  opacities[0] = 0.f;
  opacities[1] = 1.f;
  anari_cpp::unmap(d, opacityArray);

  vtkm::Float64 min = field_range.Min;
  vtkm::Float64 max = field_range.Max;
  vtkm::Int32 numPoints = color_table.GetNumberOfPoints();
  mapper.SetANARIColorMap(colorArray, opacityArray, true);
  mapper.SetANARIColorMapValueRange(vtkm::Vec2f_64(min, max));
//  mapper.SetANARIColorMapOpacityScale(0.5f);
}


} //  namespace detail

ANARITriangleRenderer::ANARITriangleRenderer()
{
  m_color_table = vtkm::cont::ColorTable("Cool to Warm");
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
  typedef vtkm::rendering::MapperRayTracer TracerType;
  m_tracer = std::make_shared<TracerType>();
  this->m_mapper = m_tracer;
  //m_tracer->SetCompositeBackground(false);
  ////
  //// add some default opacity to the color table
  ////
  //m_color_table.AddPointAlpha(0.0f, .02);
  //m_color_table.AddPointAlpha(.0f, .5);
  //m_has_unstructured = false;
}

ANARITriangleRenderer::~ANARITriangleRenderer()
{
}

void
ANARITriangleRenderer::Update()
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

void
ANARITriangleRenderer::DoExecute()
{

  // Build Scene
  vtkm::interop::anari::ANARIScene scene(m_device);
  int num_domains = m_input->GetNumberOfDomains();
  //vtkm::Range field_range = m_input->GetGlobalRange(m_field_name).ReadPortal().Get(0);
  for (int i = 0; i < num_domains; ++i)
  {

    auto& mTri = scene.AddMapper(vtkm::interop::anari::ANARIMapperTriangles(m_device));
    mTri.SetName(("triangle_" + std::to_string(i)).c_str());
    mTri.SetActor({ 
      m_input->GetDomain(i).GetCellSet(), 
      m_input->GetDomain(i).GetCoordinateSystem(), 
      m_input->GetDomain(i).GetField(m_field_name) 
    });
    mTri.SetColorTable(m_color_table);
    //vtkh::detail::setColorMap(m_device, mTri, m_color_table, field_range);
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
}


void
ANARITriangleRenderer::PreExecute()
{
  Renderer::PreExecute();
}

void
ANARITriangleRenderer::PostExecute()
{
  int total_renders = static_cast<int>(m_renders.size());
  if(m_do_composite)
  {
    this->Composite(total_renders);
  }
}


Renderer::vtkmCanvasPtr
ANARITriangleRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

std::string
ANARITriangleRenderer::GetName() const
{
  return "vtkh::ANARITriangleRenderer";
}

} // namespace vtkh
