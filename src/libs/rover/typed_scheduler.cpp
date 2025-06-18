//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// rover includes
#include "settings.hpp"
#include "vtkm_typedefs.hpp"
#include <algorithm>
#include <typed_scheduler.hpp>

using namespace conduit;

namespace rover
{

template<typename FloatType>
TypedScheduler<FloatType>::TypedScheduler()
{
  // TODO: Maybe we can just pass a RayGenerator when constructing
  // a new scheduler, since we already instantiate it beforehand
  m_ray_generator = nullptr;
}

#ifdef ROVER_PARALLEL
template<typename FloatType>
void
TypedScheduler<FloatType>::set_comm_handle(MPI_Comm comm_handle)
{
  m_comm_handle = comm_handle;
}
#endif

template<typename FloatType>
void
TypedScheduler<FloatType>::add_dataset(vtkmDataSet &dataset)
{
  ROVER_INFO("TypedScheduler::add_dataset: adding domain " << m_domains.size());
  Domain domain;
  domain.set_dataset(dataset);
  m_domains.push_back(domain);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::set_ray_generator(RayGenerator *ray_generator)
{
  m_ray_generator = ray_generator;
}

template<typename FloatType>
void
TypedScheduler<FloatType>::create_default_background(const int num_channels)
{
  // Initialize background intensities to 0.0f
  m_background.resize(num_channels, 0.0f);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::set_background(const std::vector<vtkm::Float32> &background)
{
  // TODO: See if this is done better with std::transform
  const size_t size = background.size();
  m_background.resize(size);

  for (size_t i = 0; i < size; ++i)
  {
    m_background[i] = static_cast<vtkm::Float64>(background[i]);
  }
}

template<typename FloatType>
void
TypedScheduler<FloatType>::set_background(const std::vector<vtkm::Float64> &background)
{
  m_background = background;
}

template<typename FloatType>
int
TypedScheduler<FloatType>::get_global_channels()
{
  // TODO: See if this is done better with std::max_element
  int num_channels = 1;
  for (size_t i = 0; i < m_domains.size(); ++i)
  {
    num_channels = std::max(num_channels, m_domains[i].get_num_channels());
  }

#ifdef ROVER_PARALLEL
  vtkmTimer timer;
  timer.Start();
  double time = 0;
  (void) time;
  int mpi_num_channels;
  MPI_Allreduce(&num_channels, &mpi_num_channels, 1, MPI_INT, MPI_MAX, m_comm_handle);
  num_channels = mpi_num_channels;
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("get_global_channels_all_reduce", time);
#endif

  ROVER_INFO("Global number of channels" << num_channels);
  return num_channels;
}

template<typename FloatType>
void
TypedScheduler<FloatType>::set_global_range_and_bounds()
{
  const int num_domains = m_domains.size();
  vtkmRange global_range;
  vtkm::Bounds global_bounds;
  vtkmTimer timer;
  timer.Start();
  double time = 0.0;

  for (int i = 0; i < num_domains; ++i)
  {
    vtkmRange local_range = m_domains[i].get_primary_range();
    global_range.Include(local_range);

    vtkm::Bounds local_bounds = m_domains[i].get_domain_bounds();
    global_bounds.Include(local_bounds);
  }

#ifdef ROVER_PARALLEL
  // Set global range in parallel
  double rank_min = global_range.Min;
  double rank_max = global_range.Max;
  double mpi_min;
  double mpi_max;
  MPI_Allreduce(&rank_min, &mpi_min, 1, MPI_DOUBLE, MPI_MIN, m_comm_handle);
  MPI_Allreduce(&rank_max, &mpi_max, 1, MPI_DOUBLE, MPI_MAX, m_comm_handle);
  global_range.Min = mpi_min;
  global_range.Max = mpi_max;

  // Set global bounds in parallel
  double x_min = global_bounds.X.Min;
  double x_max = global_bounds.X.Max;
  double y_min = global_bounds.Y.Min;
  double y_max = global_bounds.Y.Max;
  double z_min = global_bounds.Z.Min;
  double z_max = global_bounds.Z.Max;
  double global_x_min = 0;
  double global_x_max = 0;
  double global_y_min = 0;
  double global_y_max = 0;
  double global_z_min = 0;
  double global_z_max = 0;

  MPI_Allreduce((void *)(&x_min),
                (void *)(&global_x_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                m_comm_handle);

  MPI_Allreduce((void *)(&x_max),
                (void *)(&global_x_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                m_comm_handle);

  MPI_Allreduce((void *)(&y_min),
                (void *)(&global_y_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                m_comm_handle);

  MPI_Allreduce((void *)(&y_max),
                (void *)(&global_y_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                m_comm_handle);

  MPI_Allreduce((void *)(&z_min),
                (void *)(&global_z_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                m_comm_handle);

  MPI_Allreduce((void *)(&z_max),
                (void *)(&global_z_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                m_comm_handle);

  global_bounds.X.Min = global_x_min;
  global_bounds.X.Max = global_x_max;
  global_bounds.Y.Min = global_y_min;
  global_bounds.Y.Max = global_y_max;
  global_bounds.Z.Min = global_z_min;
  global_bounds.Z.Max = global_z_max;
#endif

  ROVER_INFO("Global range " << global_range);
  ROVER_INFO("Global bounds " << global_bounds);

  for (int i = 0; i < num_domains; ++i)
  {
    m_domains[i].set_primary_range(global_range);
    m_domains[i].set_global_bounds(global_bounds);
  }

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("set_global_range_and_bounds", time);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial)
{
  // TODO: We might be able to simplify this by using emplace_back
  PartialImage<FloatType> partial_image;
  partial_image.m_pixel_ids = partial.PixelIds;
  partial_image.m_distances = partial.Distances;
  partial_image.m_buffer = partial.Buffer;
  partial_image.m_intensities = partial.Intensities;
  m_partial_images.push_back(partial_image);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::composite()
{
  // TODO: This function has a lot of duplication that can
  // probably be improved
  int rank = 0;
#ifdef ROVER_PARALLEL
  MPI_Comm_rank(m_comm_handle, &rank);
#endif

#if 0 // removing volume renderer
  if (m_render_settings.m_render_mode == volume)
  {
    vtkh::PartialCompositor<vtkh::VolumePartial<FloatType>> compositor;
    compositor.set_background(m_background);
#ifdef ROVER_PARALLEL
    compositor.set_comm_handle(MPI_Comm_c2f(m_comm_handle));
#endif
    const int num_partials = m_partial_images.size();
    int width = m_partial_images[0].m_width;
    int height = m_partial_images[0].m_height;
    std::vector<std::vector<vtkh::VolumePartial<FloatType>>> partials;
    partials.resize(num_partials);
    for (int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::VolumePartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if (rank == 0)
    {
      // data only valid on rank = 0
      p_result.store(result,m_background, width, height);
      p_result.make_red_pixel(629, 566);
    }

    m_result = p_result;
  }
  else
  {
  }
#endif

  const std::string emission = rover::settings["rover/emission"].as_string();
  if ("" != emission)
  {
    vtkh::PartialCompositor<vtkh::EmissionPartial<FloatType>> compositor;
    compositor.set_background(m_background);
#ifdef ROVER_PARALLEL
    compositor.set_comm_handle(MPI_Comm_c2f(m_comm_handle));
#endif
    const int num_partials = m_partial_images.size();
    std::vector<std::vector<vtkh::EmissionPartial<FloatType>>> partials(num_partials);

    for (int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::EmissionPartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if (0 == rank)
    {
      // Data is only valid on rank = 0
      p_result.store(result, m_background);
    }

    m_result = p_result;
  }
  else // The case where only the absorption field is set
  {
    vtkh::PartialCompositor<vtkh::AbsorptionPartial<FloatType>> compositor;
    compositor.set_background(m_background);
#ifdef ROVER_PARALLEL
    compositor.set_comm_handle(MPI_Comm_c2f(m_comm_handle));
#endif
    const int num_partials = m_partial_images.size();
    std::vector<std::vector<vtkh::AbsorptionPartial<FloatType>>> partials;
    partials.resize(num_partials);
    for (int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::AbsorptionPartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if (0 == rank)
    {
      // data only valid on rank = 0
      p_result.store(result, m_background);
    }

    m_result = p_result;
  }
  // } // removing volume renderer
  ROVER_INFO("Schedule: compositing complete");
}

template<typename FloatType>
void
TypedScheduler<FloatType>::trace_rays()
{
  ROVER_INFO("Executing TypedScheduler::trace_rays");
  vtkmTimer tot_timer;
  vtkmTimer timer;
  tot_timer.Start();
  timer.Start();
  double time = 0.0;
  ROVER_DATA_OPEN("schedule_trace");

  if (!m_ray_generator)
  {
    throw RoverException("Error: ray generator must be set before execute is called");
  }

  m_ray_generator->reset();

  const int num_domains = static_cast<int>(m_domains.size());
  ROVER_INFO("TypedScheduler set render settings for " << num_domains << " domains");
  for (int i = 0; i < num_domains; ++i)
  {
    m_domains[i].init();
  }

  ROVER_INFO("Done TypedScheduler set render settings for " << num_domains << " domains");
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("setup", time);

  set_global_range_and_bounds();

  vtkmTimer trace_timer;
  trace_timer.Start();

  // TODO: See if we can make this loop more lightweight
  for (int i = 0; i < num_domains; ++i)
  {
    vtkmTimer domain_timer;
    domain_timer.Start();
    std::stringstream domain_s;
    domain_s << "trace_domain_" << i;
    ROVER_DATA_OPEN(domain_s.str());

    vtkmLogger::GetInstance()->Clear();

    // TODO: Don't love that we need dynamic_cast
    if (!dynamic_cast<CameraGenerator*>(m_ray_generator))
    {
      throw RoverException("Error: RayGenerator instance must be a CameraGenerator");
    }

    CameraGenerator *generator = dynamic_cast<CameraGenerator*>(m_ray_generator);
    // Setting the coordinate system miminizes the number of rays generated
    generator->set_coordinates(m_domains[i].get_dataset().GetCoordinateSystem());
    ROVER_INFO("Generating rays for domian " << i);

    timer.Start();

    vtkmRayTracing::Ray<FloatType> rays;
    m_ray_generator->get_rays(rays);

    ROVER_INFO("Generated " << rays.NumRays << " rays");
    m_domains[i].init_rays(rays);

    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("m_domains_init_rays", time);
    ROVER_INFO("Tracing domain " << i);

    timer.Start();
    std::vector<vtkmRayTracing::PartialComposite<FloatType>> partials;
    partials = m_domains[i].partial_trace(rays);
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_trace", time);

#ifdef ROVER_ENABLE_LOGGING
    DataLogger::GetInstance()->GetStream()<<vtkmLogger::GetInstance()->GetStream().str();
#endif

    ROVER_INFO("Schedule: creating partial image in domain "<<i);

    // Create a partial images from the completed rays
    for (size_t p = 0; p < partials.size(); ++p)
    {
      add_partial(partials[p]);
    }

    timer.Start();
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_push_back", time);

    time = domain_timer.GetElapsedTime();
    ROVER_DATA_CLOSE(time);
    ROVER_INFO("Schedule: done tracing domain "<<i);
  }

  timer.Start();
  time = trace_timer.GetElapsedTime();
  ROVER_DATA_ADD("total_trace", time);
  int num_channels = this->get_global_channels();

  vtkmTimer t1;
  t1.Start();

  // Add a blank partial image if we had no domains
  if (num_domains == 0 || m_partial_images.size() == 0)
  {
    PartialImage<FloatType> partial_image;
    partial_image.m_buffer =
      vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);

    const std::string emission = rover::settings["rover/emission"].as_string();
    if ("" != emission)
    {
      partial_image.m_intensities =
        vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);
    }
    m_partial_images.push_back(partial_image);
  }

  ROVER_DATA_ADD("blank_partial_image", t1.GetElapsedTime());
  t1.Start();

  if (m_background.size() == 0)
  {
    create_default_background(num_channels);
  }

  ROVER_DATA_ADD("default_bg", t1.GetElapsedTime());
  t1.Start();

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("mid", t1.GetElapsedTime());
  timer.Start();

  // Composite the results
  timer.Start();
  composite();
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("compositing", time);
  timer.Start();

  m_partial_images.clear();
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("clear", time);

  double tot_time = tot_timer.GetElapsedTime();
  ROVER_DATA_CLOSE(tot_time);

  ROVER_INFO("Finished TypedScheduler::trace_rays");
}

template<typename FloatType>
void TypedScheduler<FloatType>::save_png(std::string filename)
{
  const int64 width = rover::settings["rover/width"].to_int64();
  const int64 height = rover::settings["rover/height"].to_int64();

  // Optional params that the user may have set
  bool has_image_params = false;
  bool log_scale;
  float64 min_value;
  float64 max_value;
  
  if (rover::settings.has_child("image_params"))
  {
    // Rover's verify_params ensures that if any of these
    // are set, all of these are set
    has_image_params = true;
    min_value = rover::settings["image_params/min_value"].value();
    max_value = rover::settings["image_params/max_value"].value();
    log_scale = rover::settings["image_params/log_scale"].as_string() == "true";
  }

  ROVER_INFO("Saving .png file with output size " << width << "x" << height);
  ascent::PNGEncoder encoder;

  // if (m_render_settings.m_render_mode == energy) // removing volume renderer
  // {

  const int num_channels = m_result.get_num_channels();
  ROVER_INFO("Saving " << num_channels << " channels");
  for (int i = 0; i < num_channels; ++i)
  {
    std::stringstream sstream;
    sstream << filename << "_" << i << ".png";
    
    if (has_image_params)
    {
      m_result.normalize_intensity(i, min_value, max_value, log_scale);
    }
    else
    {
      m_result.normalize_intensity(i);
    }
    
    FloatType *buffer = get_vtkm_ptr(m_result.get_intensity(i));
    encoder.EncodeChannel(buffer, width, height);
    encoder.Save(sstream.str());
  }

#if 0 // removing volume renderer
  }
  else
  {

    assert(m_result.get_num_channels() == 4);
    vtkm::cont::ArrayHandle<FloatType> colors;
    colors = m_result.flatten_intensities();
    FloatType * buffer
      = get_vtkm_ptr(colors);

    encoder.Encode(buffer, width, height);
    encoder.Save(file_name + ".png");
  }
#endif
}

template<typename FloatType>
void TypedScheduler<FloatType>::save_bov(std::string file_name)
{
  const int64 width = rover::settings["rover/width"].to_int64();
  const int64 height = rover::settings["rover/height"].to_int64();
  const int64 size = height * width;

  ROVER_INFO("Saving bov file with output size " << width << "x" << height);
  ascent::PNGEncoder encoder;
  
  // if (m_render_settings.m_render_mode == energy) // removing volume renderer
  // {
    
  const int num_channels = m_result.get_num_channels();
  ROVER_INFO("Saving bov with " << num_channels << " channels");

  for (int i = 0; i < num_channels; ++i)
  {
    std::stringstream sstream;
    sstream << file_name  << "_" << i << ".bov";
    m_result.normalize_intensity(i);

    FloatType *buffer = get_vtkm_ptr(m_result.get_intensity(i));
    std::fstream bov(sstream.str(), std::ios::out | std::ios::binary);
    bov.write((char*)buffer, sizeof(FloatType) *size);
    bov.close();
  }
  // } // removing volume renderer
}

template<typename FloatType>
void
TypedScheduler<FloatType>::to_blueprint(Node &data)
{
  const int64 width = rover::settings["rover/width"].to_int64();
  const int64 height = rover::settings["rover/height"].to_int64();
  const int num_channels = m_result.get_num_channels();

  vtkmCamera camera = m_ray_generator->get_camera();
  vtkmVec3f position = camera.GetPosition();
  vtkmVec3f look_at = camera.GetLookAt();
  vtkmVec3f up = camera.GetViewUp();
  const double zoom = camera.GetZoom();
  const double view_angle = camera.GetFieldOfView();
  const double near_plane = camera.GetClippingRange().Min;
  const double far_plane = camera.GetClippingRange().Max;
  vtkmVec2f xy_pan = camera.GetPan();

  // Non-perspective calculations pulled from VisIt
  const double distance = vtkm::Magnitude(look_at - position);
  // TODO: Make sure that this is a valid way to compute view_height
  const double view_height = 2.0 * distance * tan((view_angle * vtkm::Pi_180()) / 2.0);
  const double view_width = (static_cast<float>(width) / static_cast<float>(height)) * view_height;
  const double near_height = view_height / zoom;
  const double near_width = view_width / zoom;
  const double far_height = near_height;
  const double far_width = near_width;
  const double detector_width = 2.0 * near_width;
  const double detector_height = 2.0 * near_height;
  const double far_detector_width = 2.0 * far_width;
  const double far_detector_height = 2.0 * far_height;
  const double near_dx = detector_width / width;
  const double near_dy = detector_height / height;

  ROVER_INFO("Saving blueprint file with output size " << width << "x" << height);

  Node &state = data["state"];
  Node &coordsets = data["coordsets"];
  Node &topologies = data["topologies"];
  Node &fields = data["fields"];

  //
  // State
  //

  if (rover::settings.has_path("state/time"))
  {
    state["time"].set(rover::settings["state/time"]);
  }

  if (rover::settings.has_path("state/cycle"))
  {
    state["cycle"].set(rover::settings["state/cycle"]);
  }
  
  Node &xray_view = state["xray_view"];
  xray_view["position"].set(&position[0], 3);
  xray_view["zoom"] = zoom;
  xray_view["look_at"].set(&look_at[0], 3);
  xray_view["up"].set(&up[0], 3);
  xray_view["fov"] = view_angle;
  xray_view["parallel_scale"] = view_height / 2.0;
  xray_view["xpan"] = xy_pan[0];
  xray_view["ypan"] = xy_pan[1];
  xray_view["near_plane"] = near_plane;
  xray_view["far_plane"] = far_plane;

  Node &xray_query = state["xray_query"];
  xray_query.set(rover::settings["rover"]);

  Node &xray_data = state["xray_data"];
  xray_data["detector_width"] = detector_width;
  xray_data["detector_height"] = detector_height;
  xray_data["intensity_max"].set(DataType::float64(1));
  xray_data["intensity_min"].set(DataType::float64(1));
  // TODO: Uncomment this when optical_depth is fixed
  // xray_data["optical_depth_max"].set(DataType::float64(1));
  // xray_data["optical_depth_min"].set(DataType::float64(1));
  xray_data["image_topo_order_of_domain_variables"] = "xyz";

  state["domain_id"] = 0;

  //
  // Image mesh
  //

  // Coordset
  Node &image_coords = coordsets["image_coords"];
  image_coords["type"] = "rectilinear";

  image_coords["values/x"].set(DataType::float64(width + 1));
  image_coords["values/y"].set(DataType::float64(height + 1));
  image_coords["values/z"].set(DataType::float64(num_channels + 1));

  float64_array image_coords_x = image_coords["values/x"].value();
  float64_array image_coords_y = image_coords["values/y"].value();
  float64_array image_coords_z = image_coords["values/z"].value();

  for (int i = 0; i <= width; i++)
  {
    image_coords_x[i] = i;
  }

  for (int i = 0; i <= height; i++)
  {
    image_coords_y[i] = i;
  }

  for (int i = 0; i <= num_channels; i++)
  {
    image_coords_z[i] = i;
  }

  image_coords["labels/x"] = "width";
  image_coords["labels/y"] = "height";
  image_coords["labels/z"] = "energy_group";

  image_coords["units/x"] = "pixels";
  image_coords["units/y"] = "pixels";
  image_coords["units/z"] = "bins";

  // if (m_render_settings.m_render_mode == energy) // removing volume renderer
  // {

  // Topology
  Node &image_topo = topologies["image_topo"];
  image_topo["coordset"] = "image_coords";
  image_topo["type"] = "rectilinear";

  if (!m_result.has_intensity(0) || !m_result.has_optical_depth(0))
  {
    ROVER_ERROR("intensity and optical depth must both be available")
  }

  // Fields
  Node &intensities = fields["intensities"];
  intensities["topology"] = "image_topo";
  intensities["association"] = "element";
  intensities["units"] = "intensity units";
  vtkm::cont::ArrayHandle<FloatType> intensity_values = m_result.flatten_intensities();
  FloatType *intensity_buffer = get_vtkm_ptr(intensity_values);
  const int num_intensity_values = intensity_values.GetNumberOfValues();

  auto intensity_min_max = std::minmax_element(intensity_buffer, intensity_buffer + num_intensity_values);
  const float64 intensity_max = *intensity_min_max.second;
  const float64 intensity_min = *intensity_min_max.first;
  xray_data["intensity_max"].set(intensity_max);
  xray_data["intensity_min"].set(intensity_min);
  
  intensities["values"].set(intensity_buffer, num_intensity_values);
  intensities["strides"].set(DataType::int64(3));
  int64_array strides = intensities["strides"].value();
  strides[0] = 1;
  strides[1] = width;
  strides[2] = width * height;

  // TODO: This seems to be broken
  Node &optical_depth = fields["optical_depth"];
  optical_depth["topology"] = "image_topo";
  optical_depth["association"] = "element";
  optical_depth["units"] = "optical depth metadata";
  vtkm::cont::ArrayHandle<FloatType> optical_values = m_result.flatten_optical_depths();
  FloatType *optical_buffer = get_vtkm_ptr(optical_values);
  const int num_optical_values = optical_values.GetNumberOfValues();

  // TODO: Uncomment this when optical_depth is fixed
  // auto optical_min_max = std::minmax_element(optical_buffer, optical_buffer + num_optical_values);
  // const float64 optical_depth_max = *optical_min_max.second;
  // const float64 optical_depth_min = *optical_min_max.first;
  // xray_data["optical_depth_max"].set(optical_depth_max);
  // xray_data["optical_depth_min"].set(optical_depth_min);

  optical_depth["values"].set(optical_buffer, num_optical_values);
  optical_depth["strides"].set(intensities["strides"]);

  //
  // Spatial mesh
  //

  // Coordset
  Node &spatial_coords = coordsets["spatial_coords"];
  spatial_coords["type"] = "rectilinear";
  spatial_coords["values/x"].set(conduit::DataType::float64(width + 1));
  spatial_coords["values/y"].set(conduit::DataType::float64(height + 1));
  spatial_coords["values/z"].set(conduit::DataType::float64(num_channels + 1));        

  float64_array spatial_coords_x = spatial_coords["values/x"].value();
  float64_array spatial_coords_y = spatial_coords["values/y"].value();
  float64_array spatial_coords_z = spatial_coords["values/z"].value();

  for (int i = 0; i <= width; i++)
  {
    spatial_coords_x[i] = i * near_dx;
  }
  
  for (int i = 0; i <= height; i++)
  {
    spatial_coords_y[i] = i * near_dy;
  }

  for (int i = 0; i <= num_channels; i++)
  {
    spatial_coords_z[i] = i;
  }  

  spatial_coords["units/x"] = "no units provided";
  spatial_coords["units/y"] = "no units provided";
  spatial_coords["units/z"] = "no units provided";

  spatial_coords["labels/x"] = "width";
  spatial_coords["labels/y"] = "height";
  spatial_coords["labels/z"] = "energy_group";

  // Topology
  Node &spatial_topo = topologies["spatial_topo"];
  spatial_topo["coordset"] = "spatial_coords";
  spatial_topo["type"] = "rectilinear";

  // Fields
  Node &intensities_spatial = fields["intensities_spatial"];
  intensities_spatial.set(intensities);
  intensities_spatial["topology"] = "spatial_topo";

  // TODO: This seems to be broken
  Node &optical_depth_spatial = fields["optical_depth_spatial"];
  optical_depth_spatial.set(optical_depth);
  optical_depth_spatial["topology"] = "spatial_topo";

  // } // removing volume renderer

  Node verify;
  if (!blueprint::verify("mesh", data, verify))
  {
    ROVER_ERROR("Error: to_blueprint failed to produce a valid conduit mesh: " << verify.to_yaml());
  }
}

// Explicit instantiation
template class TypedScheduler<vtkm::Float32>;
template class TypedScheduler<vtkm::Float64>;

}; // namespace rover
