#include "ANARIRenderer.hpp"

#include <vtkh/utils/viskores_array_utils.hpp>
#include <vtkh/compositing/Compositor.hpp>
#include <vtkh/Logger.hpp>


#include <memory>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

#include <viskores/interop/anari/ANARIScene.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/cont/ColorTable.h>
#include <viskores/rendering/ConnectivityProxy.h>
#include <viskores/rendering/raytracing/Camera.h>

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
//  bool verbose = *(bool*)userData;
//  if (!verbose)
//    return;
  (void)userData;
  //(void)device;
  (void)source;
  //(void)sourceType;
  //(void)code;

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
    : std::getenv("VISKORES_ANARI_LIBRARY")      ? std::getenv("VISKORES_ANARI_LIBRARY")
    : std::getenv("VISKORES_TEST_ANARI_LIBRARY") ? std::getenv("VISKORES_TEST_ANARI_LIBRARY") // fall back to the old environment variable
    : nullptr;
  std::cout << "Library loaded: " << (libraryName ? libraryName : "helide") << std::endl;
  static bool verbose = std::getenv("VISKORES_ANARI_VERBOSE") != nullptr;
  static bool debug = std::getenv("VISKORES_ANARI_DEBUG_DEVICE") != nullptr;
  static char* trace_dir = std::getenv("VISKORES_ANARI_DEBUG_TRACE_DIR");

  auto lib = anari_cpp::loadLibrary(libraryName ? libraryName : "helide", StatusFunc, &verbose);
  auto dev = anari_cpp::newDevice(lib, "default");
  anari_cpp::unloadLibrary(lib);

  return dev;
}

//static void setColorMap(anari_cpp::Device d, viskores::interop::anari::ANARIMapper& mapper, viskores::Range &range)
//{
//  auto colorArray = anari_cpp::newArray1D(d, ANARI_FLOAT32_VEC3, 3);
//  auto* colors = anari_cpp::map<viskores::Vec3f_32>(d, colorArray);
//  colors[0] = viskores::Vec3f_32(0.23, 0.29, 0.75);
//  colors[1] = viskores::Vec3f_32(0.87, 0.87, 0.87);
//  colors[2] = viskores::Vec3f_32(0.71, 0.02, 0.15);
////color table point: 0 rgba: 0 0.231373 0.298039 0.752941
////color table point: 1 rgba: 0.5 0.865 0.865 0.865
////color table point: 2 rgba: 1 0.705882 0.0156863 0.14902
//  anari_cpp::unmap(d, colorArray);
//
//  auto opacityArray = anari_cpp::newArray1D(d, ANARI_FLOAT32, 2);
//  auto* opacities = anari_cpp::map<float>(d, opacityArray);
//  opacities[0] = 0.f;
//  opacities[1] = 1.f;
//  anari_cpp::unmap(d, opacityArray);
//
//  float min = range.Min;
//  float max = range.Max;
//  mapper.SetANARIColorMap(colorArray, opacityArray, true);
//  mapper.SetANARIColorMapValueRange(viskores::Vec2f_32(min, max));
//  mapper.SetANARIColorMapOpacityScale(0.5f);
//}

static void setColorMap(anari_cpp::Device d, viskores::interop::anari::ANARIMapper& mapper, viskores::Range &range, viskores::cont::ColorTable &color_table)
{
  viskores::Int32 num_points = color_table.GetNumberOfPoints();
  auto colorArray = anari_cpp::newArray1D(d, ANARI_FLOAT32_VEC3, num_points);
  auto* colors = anari_cpp::map<viskores::Vec3f_32>(d, colorArray);
  for(int i = 0; i < num_points; i++)
  {
    viskores::Vec4f_64 color;
    //returned as location, RGB
    color_table.GetPoint(i,color);
    colors[i] = viskores::Vec3f_32((float)color[1], (float)color[2], (float)color[3]);
    //colors[i] = viskores::Vec3f_64(i%2, i%1, i%2);
  }
//color table point: 0 rgba: 0 0.231373 0.298039 0.752941
//color table point: 1 rgba: 0.5 0.865 0.865 0.865
//color table point: 2 rgba: 1 0.705882 0.0156863 0.14902
  anari_cpp::unmap(d, colorArray);

  auto opacityArray = anari_cpp::newArray1D(d, ANARI_FLOAT32, 2);
  auto* opacities = anari_cpp::map<float>(d, opacityArray);
  opacities[0] = 0.f;
  opacities[1] = 1.f;
  anari_cpp::unmap(d, opacityArray);

  float min = range.Min;
  float max = range.Max;
  mapper.SetANARIColorMap(colorArray, opacityArray, true);
  mapper.SetANARIColorMapValueRange(viskores::Vec2f_32(min, max));
  mapper.SetANARIColorMapOpacityScale(0.5f);
}


static void
setColorMap2(anari_cpp::Device d, viskores::interop::anari::ANARIMapper& mapper, viskores::cont::ColorTable &color_table, viskores::Range &field_range)
{
  viskores::Int32 num_points = color_table.GetNumberOfPoints();
  auto colorArray = anari_cpp::newArray1D(d, ANARI_FLOAT64_VEC3, num_points);
  auto* colors = anari_cpp::map<viskores::Vec3f_64>(d, colorArray);
  for(int i = 0; i < num_points; i++)
  {
    viskores::Vec4f_64 color;
    //returned as location, RGB
    color_table.GetPoint(i,color);
    //colors[i] = viskores::Vec3f_64(color[1], color[2], color[3]);
    //colors[i] = viskores::Vec3f_64(i%2, i%1, i%2);
  }
  colors[0] = viskores::Vec3f_32(0.f, 0.f, 1.f);
  colors[1] = viskores::Vec3f_32(0.f, 1.f, 0.f);
  colors[2] = viskores::Vec3f_32(1.f, 0.f, 0.f);

  anari_cpp::unmap(d, colorArray);

  viskores::Int32 num_points_alpha = color_table.GetNumberOfPointsAlpha();
  auto opacityArray = anari_cpp::newArray1D(d, ANARI_FLOAT64, num_points_alpha);
  auto* opacities = anari_cpp::map<float>(d, opacityArray);
  for(int i = 0; i < num_points_alpha; i++)
  {
    viskores::Vec4f_64 alpha;
    //returned as location, alpha, midpoint and sharpness
    color_table.GetPointAlpha(i,alpha);
    opacities[i] = alpha[1];
  }
  opacities[0] = 0.;
  opacities[1] = 1.;
  anari_cpp::unmap(d, opacityArray);

  viskores::Float64 min = field_range.Min;
  viskores::Float64 max = field_range.Max;
  mapper.SetANARIColorMap(colorArray, opacityArray, true);
  mapper.SetANARIColorMapValueRange(viskores::Vec2f_64(min, max));
//  mapper.SetANARIColorMapOpacityScale(0.5f);
}


} //  namespace detail

ANARIRenderer::ANARIRenderer()
{
  m_color_table = viskores::cont::ColorTable("Cool to Warm");
  //old:
  typedef viskores::rendering::MapperRayTracer TracerType;
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

ANARIRenderer::~ANARIRenderer()
{
}

void
ANARIRenderer::SetRenderers(std::vector<vtkh::ANARIRenderer*> anari_renderers)
{
  m_anari_renderers = anari_renderers;
}

void
ANARIRenderer::SetNumberOfSamples(int num_samples)
{
  m_num_samples = num_samples;
}

bool
ANARIRenderer::IsANARITriangle(ANARIRenderer *renderer)
{
  return dynamic_cast<ANARITriangleRenderer*>(renderer) != nullptr;
}

bool
ANARIRenderer::IsANARIVolume(ANARIRenderer *renderer)
{
  return dynamic_cast<ANARIVolumeRenderer*>(renderer) != nullptr;
}

bool
ANARIRenderer::IsANARIPoint(ANARIRenderer *renderer)
{
  return dynamic_cast<ANARIPointRenderer*>(renderer) != nullptr;
}

bool
ANARIRenderer::IsANARIGlyph(ANARIRenderer *renderer)
{
  return dynamic_cast<ANARIGlyphRenderer*>(renderer) != nullptr;
}

void
ANARIRenderer::Update()
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
ANARIRenderer::DoExecute()
{
  //Load Device
  m_device = detail::anari_device_load();
  m_renderer = anari_cpp::newObject<anari_cpp::Renderer>(m_device,"default");
  m_frame = anari_cpp::newObject<anari_cpp::Frame>(m_device);

  //clear lights
  for (auto& light : m_lights)
  {
    anari_cpp::release(m_device, light);
  }
  m_lights.clear();

  // create default m_lights
  anari_cpp::Light sun = anari_cpp::newObject<anari_cpp::Light>(m_device, "directional");
  anari_cpp::setParameter(m_device, sun, "direction", viskores::Vec3f_32(0.0f, -1.0f, 0.0f));
  anari_cpp::setParameter(m_device, sun, "irradiance", 2.f);
  anari_cpp::setParameter(m_device, sun, "angularDiameter", 0.00925f);
  anari_cpp::setParameter(m_device, sun, "radiance", 1.f);
  anari_cpp::commitParameters(m_device, sun);
  m_lights.push_back(sun);

  // Build Scene
  viskores::interop::anari::ANARIScene scene(m_device);
  int num_renderers = m_anari_renderers.size();
  bool isVol, isTri, isGlyph, isPoint;

  //loop through anari renderers
  //add them all to a scene
  //with their respective mappers
  //std::cerr << "num_renderers: " << num_renderers << std::endl;
  for(int i = 0; i < num_renderers; i++)
  {
    viskores::Range field_range = m_input->GetGlobalRange(m_field_name).ReadPortal().Get(0);
    int num_domains = m_input->GetNumberOfDomains();
    auto a_renderer = m_anari_renderers[i];
    //std::cerr << "for renderer i: " << i << std::endl;

    isVol = IsANARIVolume(a_renderer);
    isTri = IsANARITriangle(a_renderer);
    isPoint = IsANARIPoint(a_renderer);
    isGlyph = IsANARIGlyph(a_renderer);
    //std::cerr << "IsVol: " << isVol << std::endl;
    //std::cerr << "IsTri: " << isTri << std::endl;
    //std::cerr << "IsPoint: " << isPoint << std::endl;
    //std::cerr << "IsGlyph: " << isGlyph << std::endl;
    //loop through domains
    //add them to ANARI scene
    for (int i = 0; i < num_domains; ++i)
    {
      if(isVol)
      {
        //TODO: Unstructured does not work with helide
        //      only works with barney and visionarray

        //if unstructured, can we just cover it in triangles?
        //if(m_input->IsUnstructured())
        //{
        //  std::cerr << "in unstructured" << std::endl;
        //  auto& mIso = scene.AddMapper(viskores::interop::anari::ANARIMapperTriangles(m_device));
        //  mIso.SetName(("isosurface_" + std::to_string(i)).c_str());
        //  mIso.SetActor({
        //    m_input->GetDomain(i).GetCellSet(),
        //    m_input->GetDomain(i).GetCoordinateSystem(),
        //    m_input->GetDomain(i).GetField(m_field_name)
        //  });
        //  mIso.SetCalculateNormals(true);
        //  vtkh::detail::setColorMap(m_device, mIso, field_range, m_color_table);
        //  //detail::set_tfn(mIso,m_device,m_color_table,m_range);
        //}

        auto& mVol = scene.AddMapper(viskores::interop::anari::ANARIMapperVolume(m_device));
        mVol.SetName(("volume_" + std::to_string(i)).c_str());
        mVol.SetActor({
          m_input->GetDomain(i).GetCellSet(),
          m_input->GetDomain(i).GetCoordinateSystem(),
          m_input->GetDomain(i).GetField(m_field_name)
        });
        vtkh::detail::setColorMap(m_device, mVol, field_range, m_color_table);

      }
      else if(isTri)
      {
        auto& mTri = scene.AddMapper(viskores::interop::anari::ANARIMapperTriangles(m_device));
        mTri.SetName(("triangle_" + std::to_string(i)).c_str());
        mTri.SetActor({
          m_input->GetDomain(i).GetCellSet(),
          m_input->GetDomain(i).GetCoordinateSystem(),
          m_input->GetDomain(i).GetField(m_field_name)
        });
        vtkh::detail::setColorMap(m_device, mTri, field_range, m_color_table);
      }
      else if(isPoint)
      {
        auto& mPoint = scene.AddMapper(viskores::interop::anari::ANARIMapperPoints(m_device));
        mPoint.SetName(("points_" + std::to_string(i)).c_str());
        mPoint.SetActor({
          m_input->GetDomain(i).GetCellSet(),
          m_input->GetDomain(i).GetCoordinateSystem(),
          m_input->GetDomain(i).GetField(m_field_name)
        });
        vtkh::detail::setColorMap(m_device, mPoint, field_range, m_color_table);

      }
      else if(isGlyph)
      {
        auto& mGlyph = scene.AddMapper(viskores::interop::anari::ANARIMapperGlyphs(m_device));
        mGlyph.SetName(("glyphs_" + std::to_string(i)).c_str());
        mGlyph.SetActor({
          m_input->GetDomain(i).GetCellSet(),
          m_input->GetDomain(i).GetCoordinateSystem(),
          m_input->GetDomain(i).GetField(m_field_name)
        });
        vtkh::detail::setColorMap(m_device, mGlyph, field_range, m_color_table);

      }
      else
      {
        //TODO: Error
        std::cerr << "This ANARI Renderer is not supported yet" << std::endl;
      }
    }
  }

  int num_renders = static_cast<int>(m_renders.size());
  for(int i = 0; i < num_renders; ++i)
  {
    viskores::rendering::Camera cam = m_renders[i].GetCamera();
    viskores::rendering::Canvas &canvas = m_renders[i].GetCanvas();
    viskores::Vec4f_32 background = m_renders[i].GetBackgroundColor().Components;
    std::string img_name = m_renders[i].GetImageName();
    viskores::Float32 height = m_renders[i].GetHeight();
    viskores::Float32 width = m_renders[i].GetWidth();
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
    const auto cam_type = cam.GetMode() == viskores::rendering::Camera::Mode::ThreeD ? "perspective" : "orthographic";
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
      anari_cpp::setParameter(m_device, camera, "fov", cam.GetFieldOfView() / 180.0 * viskores::Pi());
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
    viskores::Vec2ui_32 img_size = viskores::Vec2ui_32(width,height);
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
    const auto a_colors = anari_cpp::map<viskores::Vec4f_32>(m_device, m_frame, "channel.color");
    const auto a_depths = anari_cpp::map<viskores::Float32>(m_device, m_frame, "channel.depth");

    ascent::PNGEncoder encoder;
    encoder.Encode((float *)a_colors.data, a_colors.width, a_colors.height);
    encoder.Save("encoder_image.png");
    auto v_colors = canvas.GetColorBuffer().WritePortal();
    auto v_depths = canvas.GetDepthBuffer().WritePortal();
    const float *d_pixels = anari::map<float>(m_device, m_frame, "channel.depth").data;
    int size = width*height;
    for(int pixel = 0; pixel < size; ++pixel)
    {
      int color_index = pixel*4;
      //std::cerr << "color index: " << color_index << std::endl;
      viskores::Vec4f_32 color;
      //color[0] = a_colors.data[color_index];
      //color[1] = a_colors.data[color_index+1];
      //color[2] = a_colors.data[color_index+2];
      //color[3] = a_colors.data[color_index+3];
      v_colors.Set(pixel,a_colors.data[pixel]);
      //viskores::rendering::Color color;
      //color.SetComponentFromByte(0, a_colors.data[color_index]);
      //color.SetComponentFromByte(1, a_colors.data[color_index + 1]);
      //color.SetComponentFromByte(2, a_colors.data[color_index + 2]);
      //color.SetComponentFromByte(3, a_colors.data[color_index + 3]);
      //std::cerr << "get depth" << std::endl;
      viskores::Float32 d = d_pixels[pixel];
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
ANARIRenderer::PreExecute()
{
  Renderer::PreExecute();
}

void
ANARIRenderer::PostExecute()
{
  int total_renders = static_cast<int>(m_renders.size());
  if(m_do_composite)
  {
    this->Composite(total_renders);
  }
}


Renderer::viskoresCanvasPtr
ANARIRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<viskores::rendering::CanvasRayTracer>(width, height);
}

std::string
ANARIRenderer::GetName() const
{
  return "vtkh::ANARIRenderer";
}

} // namespace vtkh
