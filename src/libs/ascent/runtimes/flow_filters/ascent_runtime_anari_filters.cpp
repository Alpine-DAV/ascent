//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_anari_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_anari_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_metadata.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_resources.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>


#if defined(ASCENT_VTKM_ENABLED)
#include <ascent_vtkh_collection.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/rendering/AutoCamera.hpp>
#include <vtkm/rendering/Camera.h>
#include <vtkm/cont/DataSet.h>

#include <vtkm/interop/anari/ANARIMapperVolume.h>
#include <vtkm/interop/anari/ANARIScene.h>

#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_vtkh_utils.hpp>
#endif

#include <png_utils/ascent_png_encoder.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#include <runtimes/ascent_data_object.hpp>

using namespace conduit;
using namespace flow;

using namespace vtkm::interop::anari;


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
loadANARIDevice()
{
  auto* libraryName = std::getenv("VTKM_TEST_ANARI_LIBRARY");
  static bool verbose = std::getenv("VTKM_TEST_ANARI_VERBOSE") != nullptr;
  auto lib = anari_cpp::loadLibrary(libraryName ? libraryName : "helide", StatusFunc, &verbose);
  auto d = anari_cpp::newDevice(lib, "default");
  anari_cpp::unloadLibrary(lib);
  return d;
}

static void 
setANARIColorMap(anari_cpp::Device d, ANARIMapper& mapper, vtkm::cont::ColorTable& tfn)
{
  constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);
  vtkm::cont::ArrayHandle<vtkm::Vec4ui_8> temp;
  {
    vtkm::cont::ScopedRuntimeDeviceTracker tracker(vtkm::cont::DeviceAdapterTagSerial{});
    tfn.Sample(1024, temp);
  }
  auto colorPortal = temp.ReadPortal();

  // Create the color and opacity arrays

  auto colorArray = anari_cpp::newArray1D(d, ANARI_FLOAT32_VEC3, 1024);
  auto* colors = anari_cpp::map<vtkm::Vec3f_32>(d, colorArray);

  auto opacityArray = anari_cpp::newArray1D(d, ANARI_FLOAT32, 1024);
  auto* opacities = anari_cpp::map<float>(d, opacityArray);

  for (vtkm::Id i = 0; i < 1024; ++i)
  {
    auto color = colorPortal.Get(i);
    colors[i] = vtkm::Vec3f_32(color[0], color[1], color[2]) * conversionToFloatSpace;
    opacities[i] = color[3] * conversionToFloatSpace;
  }

  anari_cpp::unmap(d, colorArray);
  anari_cpp::unmap(d, opacityArray);

  auto range = tfn.GetRange();

  mapper.SetANARIColorMap(colorArray, opacityArray, true);
  mapper.SetANARIColorMapValueRange(vtkm::Vec2f_32(range.Min, range.Max));
  mapper.SetANARIColorMapOpacityScale(0.5f);
}

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

namespace detail
{

/* defined in ascent_runtime_anari_filters.cpp */
std::string
check_color_table_surprises(const conduit::Node &color_table);

static bool 
check_image_names(const conduit::Node &params, conduit::Node &info)
{
  bool res = true;
  if (!params.has_path("image_prefix") && !params.has_path("camera/db_name"))
  {
    res = false;
    info.append() = "Anari ray rendering paths must include either "
                    "a 'image_prefix' (if its a single image) or a "
                    "'camera/db_name' (if using a cinema camere)";
  }
  if (params.has_path("image_prefix") && params.has_path("camera/db_name"))
  {
    res = false;
    info.append() = "Anari ray rendering paths cannot use both "
                    "a 'image_prefix' (if its a single image) and a "
                    "'camera/db_name' (if using a cinema camere)";
  }
  return res;
}

}; // namespace detail

//-----------------------------------------------------------------------------
struct AnariVolume::Impl {
public:
  anari_cpp::Device device;
  anari_cpp::Renderer renderer;
  anari_cpp::Camera camera;
  anari_cpp::Frame frame;

  vtkm::cont::ColorTable tfn = vtkm::cont::ColorTable("Cool to Warm");

  // camera parameters
  vtkm::Vec3f_32 cam_pos = vtkm::Vec3f_32(800, 800, 800);
  vtkm::Vec3f_32 cam_dir = vtkm::Vec3f_32(-1,-1,-1);
  vtkm::Vec3f_32 cam_up  = vtkm::Vec3f_32(0,1,0);

  // framebuffer parameters
  std::string img_name = "interop_anari_volume";
  vtkm::Vec2ui_32 img_size = vtkm::Vec2ui_32(1024, 768);

  // renderer parameters
  vtkm::Vec4f_32 background = vtkm::Vec4f_32(0.3f, 0.3f, 0.3f, 1.f);
  int pixelSamples = 64;

public:
  ~Impl();
  Impl();
  void render(const vtkm::cont::DataSet &dset);
};

AnariVolume::Impl::Impl()
{
  device = loadANARIDevice();
  renderer = anari_cpp::newObject<anari_cpp::Renderer>(device, "default");
  camera = anari_cpp::newObject<anari_cpp::Camera>(device, "perspective");
  frame = anari_cpp::newObject<anari_cpp::Frame>(device);
}

AnariVolume::Impl::~Impl()
{
  anari_cpp::release(device, camera);
  anari_cpp::release(device, renderer);
  anari_cpp::release(device, frame);
  anari_cpp::release(device, device);
}

void
AnariVolume::Impl::render(const vtkm::cont::DataSet &dset)
{
  // scene graph parameters
  ANARIScene scene(device);

  auto& mVol = scene.AddMapper(vtkm::interop::anari::ANARIMapperVolume(device));
  mVol.SetName("volume");
  mVol.SetActor({ dset.GetCellSet(), dset.GetCoordinateSystem(), dset.GetField("energy") });
  setANARIColorMap(device, mVol, tfn);

  // renderer parameters
  anari_cpp::setParameter(device, renderer, "background", background);
  anari_cpp::setParameter(device, renderer, "pixelSamples", pixelSamples);
  anari_cpp::commitParameters(device, renderer);

  // camera parameters
  anari_cpp::setParameter(device, camera, "aspect", img_size[0] / float(img_size[1]));
  anari_cpp::setParameter(device, camera, "position",  cam_pos);
  anari_cpp::setParameter(device, camera, "direction", cam_dir);
  anari_cpp::setParameter(device, camera, "up",        cam_up);
  anari_cpp::commitParameters(device, camera);

  // frame parameters
  anari_cpp::setParameter(device, frame, "size", img_size);
  anari_cpp::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari_cpp::setParameter(device, frame, "world", scene.GetANARIWorld());
  anari_cpp::setParameter(device, frame, "camera", camera);
  anari_cpp::setParameter(device, frame, "renderer", renderer);
  anari_cpp::commitParameters(device, frame);

  // render and wait for completion
  anari_cpp::render(device, frame);
  anari_cpp::wait(device, frame);

  // on rank 0, access framebuffer and write its content as PNG file
  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);
#endif
  if (rank == 0) 
  {
    const auto fb = anari_cpp::map<vtkm::Vec4f_32>(device, frame, "channel.color");
    ascent::PNGEncoder encoder;
    encoder.Encode((float*)fb.data, fb.width, fb.height);
    encoder.Save(img_name  + ".png");
    anari_cpp::unmap(device, frame, "channel.color");
  }
}

//-----------------------------------------------------------------------------
AnariVolume::AnariVolume()
  : Filter(), pimpl(new AnariVolume::Impl())
{
  // empty
}

//-----------------------------------------------------------------------------
AnariVolume::~AnariVolume()
{
  // empty
}

//-----------------------------------------------------------------------------
void
AnariVolume::declare_interface(Node &i)
{
  i["type_name"]   = "dray_volume";
  i["port_names"].append() = "in";
  i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
AnariVolume::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();

  bool res = true;

  res &= check_string("field",params, info, true);
  res &= detail::check_image_names(params, info);
  res &= check_numeric("min_value",params, info, false);
  res &= check_numeric("max_value",params, info, false);
  res &= check_numeric("image_width",params, info, false);
  res &= check_numeric("image_height",params, info, false);
  res &= check_string("log_scale",params, info, false);
  // res &= check_string("annotations",params, info, false);

  std::vector<std::string> valid_paths;
  std::vector<std::string> ignore_paths;

  valid_paths.push_back("field");
  valid_paths.push_back("image_prefix");
  valid_paths.push_back("min_value");
  valid_paths.push_back("max_value");
  valid_paths.push_back("image_width");
  valid_paths.push_back("image_height");
  valid_paths.push_back("log_scale");
  valid_paths.push_back("annotations");

  valid_paths.push_back("camera/look_at");
  valid_paths.push_back("camera/position");
  valid_paths.push_back("camera/up");
  valid_paths.push_back("camera/fov");
  valid_paths.push_back("camera/xpan");
  valid_paths.push_back("camera/ypan");
  valid_paths.push_back("camera/zoom");
  valid_paths.push_back("camera/near_plane");
  valid_paths.push_back("camera/far_plane");
  valid_paths.push_back("camera/azimuth");
  valid_paths.push_back("camera/elevation");

  // filter knobs
  res &= check_numeric("samples",params, info, false);
  res &= check_string("use_lighing",params, info, false);

  valid_paths.push_back("samples");
  valid_paths.push_back("use_lighting");

  ignore_paths.push_back("color_table");

  std::string surprises = surprise_check(valid_paths, ignore_paths, params);

  if (params.has_path("color_table"))
  {
    surprises += detail::check_color_table_surprises(params["color_table"]);
  }

  if (surprises != "")
  {
    res = false;
    info["errors"].append() = surprises;
  }
  return res;
}

//-----------------------------------------------------------------------------
void
AnariVolume::execute()
{
  if (!input(0).check_type<DataObject>())
  {
    ASCENT_ERROR("Anari Volume input must be a DataObject");
  }

  DataObject *d_input = input<DataObject>(0);
  if (!d_input->is_valid())
  {
    return;
  }

  // Parse input data as a VTK-h collection
  VTKHCollection *collection = d_input->as_vtkh_collection().get();
  std::vector<std::string> topos = collection->topology_names();

  if (topos.size() != 1)
  {
    ASCENT_ERROR("Anari Volume accepts only one topology");
  }
  auto &topo = collection->dataset_by_topology(topos[0]);

  if (topo.GetNumberOfDomains() != 1)
  {
    ASCENT_ERROR("Anari Volume accepts only one domain");
  }
  auto& dset = topo.GetDomain(0);
  // dset.PrintSummary(std::cout);

  // It is important to compute the data bounds
  vtkm::Bounds bounds = collection->global_bounds();

  // Parse camera
  vtkm::rendering::Camera camera;
  if (params().has_path("camera"))
  {
    parse_camera(params()["camera"], camera);
  }
  else // if we don't have camera params, we need to add a default camera
  {
    camera.ResetToBounds(bounds);
  }    
  pimpl->cam_pos = camera.GetPosition();
  pimpl->cam_dir = camera.GetLookAt() - camera.GetPosition();
  pimpl->cam_up  = camera.GetViewUp();

  // Set transfer function
  if(params().has_path("color_table"))
  {
    vtkm::cont::ColorTable color_table = parse_color_table(params()["color_table"]);
    // if (type != "mesh")
    // {
    //   if (params()["color_table"].has_path("annotation"))
    //   {
    //      if (params()["color_table/annotation"].as_string() == "false")
    //      {
    //         renderer->DisableColorBar();
    //      }
    //   }
    // }
    pimpl->tfn = color_table;
  }

  vtkm::Range scalar_range;
  if (params().has_path("min_value"))
  {
    scalar_range.Min = params()["min_value"].to_float64();
  }
  if (params().has_path("max_value"))
  {
    scalar_range.Max = params()["max_value"].to_float64();
  }
  // renderer->SetRange(scalar_range);

  // Initialize ANARI /////////////////////////////////////////////////////////
  pimpl->render(dset);
  // Finalize ANARI ///////////////////////////////////////////////////////////
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
