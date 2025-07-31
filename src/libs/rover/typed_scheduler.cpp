//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// rover includes
#include "ray_generators/ray_generator.hpp"
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
  m_num_local_domains = 0;
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
TypedScheduler<FloatType>::add_dataset(vtkh::DataSet &dataset)
{
  m_num_local_domains = dataset.GetNumberOfDomains();
  m_domains.reserve(m_num_local_domains);
  for (int i = 0; i < m_num_local_domains; i++)
  {
    ROVER_INFO("TypedScheduler::add_dataset: adding domain " << i);
    m_domains.emplace_back(dataset.GetDomain(i));
  }
}

template<typename FloatType>
void
TypedScheduler<FloatType>::set_ray_generator(RayGenerator *ray_generator)
{
  m_ray_generator = ray_generator;
}

template<typename FloatType>
void
TypedScheduler<FloatType>::create_background(const int num_channels)
{
  // Initialize background intensities to 0.0f (by default)
  const float64 background_intensity = rover::settings["background_intensity"].to_float64();
  m_background.resize(num_channels, background_intensity);
}

template<typename FloatType>
int
TypedScheduler<FloatType>::get_global_channels()
{
  int num_channels = 1;
  for (auto& domain : m_domains)
  {
    num_channels = std::max(num_channels, domain.get_num_channels());
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
  vtkmRange global_range;
  vtkm::Bounds global_bounds;
  vtkmTimer timer;
  timer.Start();
  double time = 0.0;

  for (int i = 0; i < m_num_local_domains; ++i)
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

  for (auto& domain : m_domains)
  {
    domain.set_primary_range(global_range);
    domain.set_global_bounds(global_bounds);
  }

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("set_global_range_and_bounds", time);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::add_partial(const vtkhRayTracing::PartialComposite<FloatType> &partial)
{
  // TODO: We might be able to simplify this by using emplace_back
  PartialImage<FloatType> partial_image;
  partial_image.m_pixel_ids = partial.PixelIds;
  partial_image.m_distances = partial.Distances;
  partial_image.m_transmission = partial.Transmission;
  partial_image.m_intensity = partial.Intensity;
  partial_image.m_optical_depth = partial.OpticalDepth;
  m_partial_images.push_back(partial_image);
}

template<typename FloatType>
void
TypedScheduler<FloatType>::composite()
{
  // TODO: Combine AbsorptionPartial and EmissionPartial
  const std::string emission = rover::settings["emission"].as_string();
  if (emission.empty())
  {
    typed_composite<vtkh::AbsorptionPartial<FloatType>>();
  }
  else // (!emission.empty())
  {
    typed_composite<vtkh::EmissionPartial<FloatType>>();
  }
  ROVER_INFO("Schedule: compositing complete");
}

template<typename FloatType>
template<typename PartialType>
void
TypedScheduler<FloatType>::typed_composite()
{
  int rank = 0;
#ifdef ROVER_PARALLEL
  MPI_Comm_rank(m_comm_handle, &rank);
#endif

  vtkh::PartialCompositor<PartialType> compositor;
#ifdef ROVER_PARALLEL
  compositor.set_comm_handle(MPI_Comm_c2f(m_comm_handle));
#endif
  compositor.set_background(m_background);

  const int num_partials = m_partial_images.size();
  std::vector<std::vector<PartialType>> partials(num_partials);

  for (int i = 0; i < num_partials; ++i)
  {
    m_partial_images[i].extract_partials(partials[i]);
  }

  std::vector<PartialType> result;
  compositor.composite(partials, result);

  // The compositor output is only on rank 0
  if (0 == rank)
  {
    PartialImage<FloatType> p_result;
    p_result.store(result, m_background);
    m_result = p_result;
  }

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
  }
#endif
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

  set_global_range_and_bounds();

  vtkmTimer trace_timer;
  trace_timer.Start();

  vtkmRayTracing::Ray<FloatType> rays;

  for (int i = 0; i < m_num_local_domains; i++)
  {
    vtkmTimer domain_timer;
    domain_timer.Start();
    std::stringstream domain_s;
    domain_s << "trace_domain_" << i;
    ROVER_DATA_OPEN(domain_s.str());

    vtkmLogger::GetInstance()->Clear();

    // Setting the coordinate system miminizes the number of rays generated
    m_ray_generator->set_coordinates(m_domains[i].get_dataset().GetCoordinateSystem());
    ROVER_INFO("Generating rays for domian " << i);

    timer.Start();

    // TODO: I'm curious about which conditions can cause rays to fail to be created
    if (!m_ray_generator->get_rays(rays))
    {
      ROVER_ERROR("Failed to create new rays");
    }

    ROVER_INFO("Generated " << rays.NumRays << " rays");
    m_domains[i].init_rays(rays);

    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("m_domains_init_rays", time);
    ROVER_INFO("Tracing domain " << i);

    timer.Start();
    std::vector<vtkhRayTracing::PartialComposite<FloatType>> partials;
    m_domains[i].partial_trace(rays, partials);
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_trace", time);

#ifdef ROVER_ENABLE_LOGGING
    DataLogger::GetInstance()->GetStream()<<vtkmLogger::GetInstance()->GetStream().str();
#endif

    ROVER_INFO("Schedule: creating partial image in domain "<<i);

    // Create a partial image from the completed rays
    for (const auto& partial : partials)
    {
      add_partial(partial);
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
  int num_channels = get_global_channels();

  vtkmTimer t1;
  t1.Start();

  // Add a blank partial image if we had no domains
  if (m_num_local_domains == 0 || m_partial_images.empty())
  {
    PartialImage<FloatType> partial_image;
    partial_image.m_transmission =
      vtkmRayTracing::ChannelBuffer<FloatType>(num_channels, 0);

    const std::string emission = rover::settings["emission"].as_string();
    if (!emission.empty())
    {
      partial_image.m_intensity =
        vtkmRayTracing::ChannelBuffer<FloatType>(num_channels, 0);
    }
    m_partial_images.push_back(partial_image);
  }

  ROVER_DATA_ADD("blank_partial_image", t1.GetElapsedTime());
  t1.Start();

  if (m_background.empty())
  {
    create_background(num_channels);
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
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();

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
    else // (!has_image_params)
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
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
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
TypedScheduler<FloatType>::write_blueprint_imaging_plane(Node &data_out,
                                                         const std::string plane_name,
                                                         const double plane_width,
                                                         const double plane_height,
                                                         const vtkmVec3f &center,
                                                         const vtkmVec3f &left,
                                                         const vtkmVec3f &up,
                                                         vtkmVec3f &llc,
                                                         vtkmVec3f &lrc,
                                                         vtkmVec3f &ulc,
                                                         vtkmVec3f &urc)
{
  // Define imaging plane coordset
  Node &plane_coords = data_out["coordsets"][plane_name + "_coords"];
  plane_coords["type"] = "explicit";
  plane_coords["values/x"].set(DataType::float64(4));
  plane_coords["values/y"].set(DataType::float64(4));
  plane_coords["values/z"].set(DataType::float64(4));
  float64_array xvals = plane_coords["values/x"].value();
  float64_array yvals = plane_coords["values/y"].value();
  float64_array zvals = plane_coords["values/z"].value();

  vtkmVec3f up_scaled = plane_height * up;
  vtkmVec3f left_scaled = plane_width * left;

  llc = center - up_scaled + left_scaled;
  lrc = center - up_scaled - left_scaled;
  ulc = center + up_scaled + left_scaled;
  urc = center + up_scaled - left_scaled;

  // Set x values
  xvals[0] = llc[0];
  xvals[1] = lrc[0];
  xvals[2] = urc[0];
  xvals[3] = ulc[0];

  // Set y values
  yvals[0] = llc[1];
  yvals[1] = lrc[1];
  yvals[2] = urc[1];
  yvals[3] = ulc[1];

  // Set z values
  zvals[0] = llc[2];
  zvals[1] = lrc[2];
  zvals[2] = urc[2];
  zvals[3] = ulc[2];

  // Define imaging plane topology
  Node &plane_topo = data_out["topologies"][plane_name + "_topo"];
  plane_topo["type"] = "unstructured";
  plane_topo["coordset"] = plane_name + "_coords";
  plane_topo["elements/shape"] = "quad";
  plane_topo["elements/connectivity"].set(DataType::int32(4));
  int32_array connectivity = plane_topo["elements/connectivity"].value();

  // Set connectivity values
  for (int i = 0; i < 4; i ++)
  {
    connectivity[i] = i;
  }

  // Define imaging plane field
  Node &plane_field = data_out["fields"][plane_name + "_field"];
  plane_field["topology"] = plane_name + "_topo";
  plane_field["association"] = "element";
  plane_field["volume_dependent"] = "false";
  plane_field["values"].set(DataType::float64(1));
  float64_array field_vals = plane_field["values"].value();
  field_vals[0] = 0;
}

template<typename FloatType>
void
TypedScheduler<FloatType>::write_blueprint_ray_corners_mesh(Node &data_out,
                                                            const vtkmVec3f &llc_near,
                                                            const vtkmVec3f &llc_far,
                                                            const vtkmVec3f &lrc_near,
                                                            const vtkmVec3f &lrc_far,
                                                            const vtkmVec3f &urc_near,
                                                            const vtkmVec3f &urc_far,
                                                            const vtkmVec3f &ulc_near,
                                                            const vtkmVec3f &ulc_far)
{
  const int num_corners = 4;
  const int num_points = 8;

  // Define ray corners coordset
  Node &ray_corners_coords = data_out["coordsets"]["ray_corners_coords"];
  ray_corners_coords["type"] = "explicit";
  ray_corners_coords["values/x"].set(DataType::float64(num_points));
  ray_corners_coords["values/y"].set(DataType::float64(num_points));
  ray_corners_coords["values/z"].set(DataType::float64(num_points));
  float64_array xvals_ray = ray_corners_coords["values/x"].value();
  float64_array yvals_ray = ray_corners_coords["values/y"].value();
  float64_array zvals_ray = ray_corners_coords["values/z"].value();

  // Set x values
  xvals_ray[0] = llc_near[0];
  xvals_ray[1] = llc_far[0];
  xvals_ray[2] = lrc_near[0];
  xvals_ray[3] = lrc_far[0];
  xvals_ray[4] = urc_near[0];
  xvals_ray[5] = urc_far[0];
  xvals_ray[6] = ulc_near[0];
  xvals_ray[7] = ulc_far[0];

  // Set y values
  yvals_ray[0] = llc_near[1];
  yvals_ray[1] = llc_far[1];
  yvals_ray[2] = lrc_near[1];
  yvals_ray[3] = lrc_far[1];
  yvals_ray[4] = urc_near[1];
  yvals_ray[5] = urc_far[1];
  yvals_ray[6] = ulc_near[1];
  yvals_ray[7] = ulc_far[1];

  // Set z values
  zvals_ray[0] = llc_near[2];
  zvals_ray[1] = llc_far[2];
  zvals_ray[2] = lrc_near[2];
  zvals_ray[3] = lrc_far[2];
  zvals_ray[4] = urc_near[2];
  zvals_ray[5] = urc_far[2];
  zvals_ray[6] = ulc_near[2];
  zvals_ray[7] = ulc_far[2];

  // Define ray corners topology
  Node &ray_corners_topo = data_out["topologies"]["ray_corners_topo"];
  ray_corners_topo["type"] = "unstructured";
  ray_corners_topo["coordset"] = "ray_corners_coords";
  ray_corners_topo["elements/shape"] = "line";
  ray_corners_topo["elements/connectivity"].set(DataType::int32(num_points));
  int32_array connectivity = ray_corners_topo["elements/connectivity"].value();

  // Set connectivity values
  for (int i = 0; i < num_points; i++)
  {
    connectivity[i] = i;
  }

  // Define ray corners field
  Node &ray_corners_field = data_out["fields"]["ray_corners_field"];
  ray_corners_field["topology"] = "ray_corners_topo";
  ray_corners_field["association"] = "element";
  ray_corners_field["volume_dependent"] = "false";
  ray_corners_field["values"].set(DataType::float64(num_corners));
  float64_array field_vals = ray_corners_field["values"].value();

  for (int i = 0; i < num_corners; i++)
  {
    field_vals[i] = 0;
  }
}

template<typename FloatType>
void
TypedScheduler<FloatType>::write_blueprint_rays_mesh(Node &data_out,
                                                     const int64 image_width,
                                                     const int64 image_height,
                                                     const double detector_width,
                                                     const double detector_height,
                                                     const vtkmVec3f &lrc_near,
                                                     const double far_detector_width,
                                                     const double far_detector_height,
                                                     const vtkmVec3f &lrc_far,
                                                     const vtkmVec3f &left,
                                                     const vtkmVec3f &up)
{
  const int64 num_lines = image_width * image_height;
  const int64 num_points = num_lines * 2;

  // Define rays coordset
  Node &ray_coords = data_out["coordsets"]["ray_coords"];
  ray_coords["type"] = "explicit";
  ray_coords["values/x"].set(DataType::float64(num_points));
  ray_coords["values/y"].set(DataType::float64(num_points));
  ray_coords["values/z"].set(DataType::float64(num_points));
  float64_array xvals_ray = ray_coords["values/x"].value();
  float64_array yvals_ray = ray_coords["values/y"].value();
  float64_array zvals_ray = ray_coords["values/z"].value();

  vtkmVec3f scaled_unit_left;
  vtkmVec3f scaled_unit_up;
  vtkmVec3f lrc;

  for (int i = 0; i < 2; i++)
  {
      double dx;
      double dy;
      if (0 == i) // 1st iteration is for the near plane
      {
          dx = detector_width / image_width;
          dy = detector_height / image_height;
          lrc = lrc_near;
      }
      else // 2nd iteration is for the far plane
      {
          dx = far_detector_width / image_width;
          dy = far_detector_height / image_height;
          lrc = lrc_far;
      }

      scaled_unit_left = dx * vtkm::Normal(left);
      scaled_unit_up = dy * vtkm::Normal(up);

      for (int j = 0; j < image_width; j++)
      {
          for (int k = 0; k < image_height; k++)
          {
              vtkmVec3f temp = lrc + (j + 0.5f) * scaled_unit_left + (k + 0.5f) * scaled_unit_up;
              // 3d to 1d conversion
              const int64 index = i * image_width * image_height + j * image_height + k;
              xvals_ray[index] = temp[0];
              yvals_ray[index] = temp[1];
              zvals_ray[index] = temp[2];
          }
      }
  }

  // Define rays topology
  Node &ray_topo = data_out["topologies"]["ray_topo"];
  ray_topo["type"] = "unstructured";
  ray_topo["coordset"] = "ray_coords";
  ray_topo["elements/shape"] = "line";
  ray_topo["elements/connectivity"].set(DataType::int32(num_points));
  int32_array connectivity = ray_topo["elements/connectivity"].value();

  // Set connectivity values
  for (int i = 0; i < num_lines; i++)
  {
    // Connect each point in the near plane to a point in the far plane
    connectivity[i * 2] = i;
    connectivity[i * 2 + 1] = i + num_lines;
  }

  // Define rays field
  Node &ray_field = data_out["fields"]["ray_field"];
  ray_field["topology"] = "ray_topo";
  ray_field["association"] = "element";
  ray_field["volume_dependent"] = "false";
  ray_field["values"].set(DataType::float64(num_lines));
  float64_array field_vals = ray_field["values"].value();

  for (int i = 0; i < num_lines; i++)
  {
    field_vals[i] = i;
  }
}

template<typename FloatType>
void
TypedScheduler<FloatType>::to_blueprint(Node &data)
{
  const int64 image_width = rover::settings["width"].to_int64();
  const int64 image_height = rover::settings["height"].to_int64();
  const double aspect_ratio = static_cast<double>(image_width) / static_cast<double>(image_height);

  const int num_channels = m_result.get_num_channels();

  vtkmCamera camera = m_ray_generator->get_camera();
  const vtkmVec3f position = camera.GetPosition();
  const vtkmVec3f look_at = camera.GetLookAt();
  const vtkmVec3f up = vtkm::Normal(camera.GetViewUp());
  const vtkmVec3f forward = vtkm::Normal(look_at - position);
  const vtkmVec3f left = vtkm::Normal(vtkm::Cross(up, forward));
  const double view_distance = vtkm::Magnitude(look_at - position);
  const double zoom = camera.GetZoom();
  const double view_angle_deg = camera.GetFieldOfView();
  const double view_angle_rad = view_angle_deg * vtkm::Pi_180f();
  const double frustum_scale = vtkm::Tan(view_angle_rad / 2.0f) / zoom;
  const double near_plane = camera.GetClippingRange().Min;
  const double far_plane = camera.GetClippingRange().Max;
  vtkmVec2f xy_pan = camera.GetPan();

  // These calculations are based on VisIt's perspective calculations, but diverge
  // due to differences in how near and far planes are represented. VisIt represents
  // them as offsets from the view plane (e.g. -0.5, 0.5), while vtkm represents them
  // as positive distances away from the camera position (e.g. 3, 300).

  const double near_height = near_plane * frustum_scale;
  const double near_width = near_height * aspect_ratio;
  const double view_height = view_distance * frustum_scale;
  const double view_width = view_height * aspect_ratio;
  const double far_height = far_plane * frustum_scale;
  const double far_width = far_height * aspect_ratio;

  const vtkmVec3f center_near = position + near_plane * forward;
  const vtkmVec3f center_view = look_at;
  const vtkmVec3f center_far = position + far_plane * forward;

  const double detector_height = near_height * 2.0f;
  const double detector_width = detector_height * aspect_ratio;
  const double far_detector_height = far_height * 2.0f;
  const double far_detector_width = far_detector_height * aspect_ratio;

  // The spatial meshes should have the same dimensions as the near plane
  const double spatial_dx = near_width * 2.0f / image_width;
  const double spatial_dy = near_height * 2.0f / image_height;

  vtkmVec3f llc_near;
  vtkmVec3f lrc_near;
  vtkmVec3f ulc_near;
  vtkmVec3f urc_near;

  vtkmVec3f llc_view;
  vtkmVec3f lrc_view;
  vtkmVec3f ulc_view;
  vtkmVec3f urc_view;

  vtkmVec3f llc_far;
  vtkmVec3f lrc_far;
  vtkmVec3f ulc_far;
  vtkmVec3f urc_far;

  ROVER_INFO("Saving blueprint file with output size " << width << "x" << height);

  Node &state = data["state"];
  Node &coordsets = data["coordsets"];
  Node &topologies = data["topologies"];
  Node &fields = data["fields"];

  //
  // State
  //

  if (rover::metadata.has_child("time"))
  {
    state["time"].set(rover::metadata["time"]);
  }

  if (rover::metadata.has_path("cycle"))
  {
    state["cycle"].set(rover::metadata["cycle"]);
  }
  
  Node &xray_view = state["xray_view"];
  xray_view["position"].set(&position[0], 3);
  xray_view["zoom"] = zoom;
  xray_view["look_at"].set(&look_at[0], 3);
  xray_view["up"].set(&up[0], 3);
  // xray_view["normal"].set(&forward[0], 3);
  xray_view["fov"] = view_angle_deg;
  // TODO: Bring this back once 2D camera support is added
  // xray_view["parallel_scale"] = view_height; // TODO: Needs validation against VisIt
  xray_view["xpan"] = xy_pan[0];
  xray_view["ypan"] = xy_pan[1];
  xray_view["near_plane"] = near_plane;
  xray_view["far_plane"] = far_plane;

  Node &xray_query = state["xray_query"];
  xray_query.set(rover::settings);

  Node &xray_data = state["xray_data"];
  xray_data["detector_width"] = detector_width; // TODO: Needs validation against VisIt
  xray_data["detector_height"] = detector_height; // TODO: Needs validation against VisIt
  xray_data["intensity_max"];
  xray_data["intensity_min"];
  xray_data["optical_depth_max"];
  xray_data["optical_depth_min"];
  xray_data["image_topo_order_of_domain_variables"] = "xyz";

  state["domain_id"] = 0;

  //
  // Image mesh
  //

  // Coordset
  Node &image_coords = coordsets["image_coords"];
  image_coords["type"] = "rectilinear";

  image_coords["values/x"].set(DataType::float64(image_width + 1));
  image_coords["values/y"].set(DataType::float64(image_height + 1));
  image_coords["values/z"].set(DataType::float64(num_channels + 1));

  float64_array image_coords_x = image_coords["values/x"].value();
  float64_array image_coords_y = image_coords["values/y"].value();
  float64_array image_coords_z = image_coords["values/z"].value();

  for (int i = 0; i <= image_width; i++)
  {
    image_coords_x[i] = i;
  }

  for (int i = 0; i <= image_height; i++)
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

  // Topology
  Node &image_topo = topologies["image_topo"];
  image_topo["coordset"] = "image_coords";
  image_topo["type"] = "rectilinear";

  // Image field
  Node &optical_depth = fields["optical_depth"];
  optical_depth["topology"] = "image_topo";
  optical_depth["association"] = "element";
  optical_depth["units"] = "optical depth metadata";
  vtkm::cont::ArrayHandle<FloatType> optical_values = m_result.flatten_optical_depth_values();
  FloatType *optical_buffer = get_vtkm_ptr(optical_values);
  const int num_optical_values = optical_values.GetNumberOfValues();

  auto optical_min_max = std::minmax_element(optical_buffer, optical_buffer + num_optical_values);
  xray_data["optical_depth_max"].set(optical_min_max.second);
  xray_data["optical_depth_min"].set(optical_min_max.first);

  optical_depth["values"].set(optical_buffer, num_optical_values);
  optical_depth["strides"].set(DataType::int64(3));
  int64_array strides = optical_depth["strides"].value();
  strides[0] = 1;
  strides[1] = image_width;
  strides[2] = image_width * image_height;

  // Spatial field
  Node &optical_depth_spatial = fields["optical_depth_spatial"];
  optical_depth_spatial.set(optical_depth);
  optical_depth_spatial["topology"] = "spatial_topo";

  // Intensity is only available in the absorption + emission case
  const std::string emission = rover::settings["emission"].as_string();
  if (!emission.empty())
  {
    // Image field
    Node &intensities = fields["intensities"];
    intensities["topology"] = "image_topo";
    intensities["association"] = "element";
    intensities["units"] = "intensity units";
    vtkm::cont::ArrayHandle<FloatType> intensity_values = m_result.flatten_intensity_values();
    FloatType *intensity_buffer = get_vtkm_ptr(intensity_values);
    const int num_intensity_values = intensity_values.GetNumberOfValues();
  
    auto intensity_min_max = std::minmax_element(intensity_buffer, intensity_buffer + num_intensity_values);
    xray_data["intensity_max"].set(intensity_min_max.second);
    xray_data["intensity_min"].set(intensity_min_max.first);
    
    intensities["values"].set(intensity_buffer, num_intensity_values);
    intensities["strides"].set(strides);

    // Spatial field
    Node &intensities_spatial = fields["intensities_spatial"];
    intensities_spatial.set(intensities);
    intensities_spatial["topology"] = "spatial_topo";
  }

  //
  // Spatial mesh
  //

  // Coordset
  Node &spatial_coords = coordsets["spatial_coords"];
  spatial_coords["type"] = "rectilinear";
  spatial_coords["values/x"].set(DataType::float64(image_width + 1));
  spatial_coords["values/y"].set(DataType::float64(image_height + 1));
  spatial_coords["values/z"].set(DataType::float64(num_channels + 1));

  float64_array spatial_coords_x = spatial_coords["values/x"].value();
  float64_array spatial_coords_y = spatial_coords["values/y"].value();
  float64_array spatial_coords_z = spatial_coords["values/z"].value();

  for (int i = 0; i <= image_width; i++)
  {
    spatial_coords_x[i] = i * spatial_dx;
  }
  
  for (int i = 0; i <= image_height; i++)
  {
    spatial_coords_y[i] = i * spatial_dy;
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

  //
  // Near plane mesh
  //

  write_blueprint_imaging_plane(data,
                                "near_plane",
                                near_width,
                                near_height,
                                center_near,
                                left,
                                up,
                                llc_near,
                                lrc_near,
                                ulc_near,
                                urc_near);

  //
  // View plane mesh
  //

  write_blueprint_imaging_plane(data,
                                "view_plane",
                                view_width,
                                view_height,
                                center_view,
                                left,
                                up,
                                llc_view,
                                lrc_view,
                                ulc_view,
                                urc_view);

  //
  // Far plane mesh
  //

  write_blueprint_imaging_plane(data,
                                "far_plane",
                                far_width,
                                far_height,
                                center_far,
                                left,
                                up,
                                llc_far,
                                lrc_far,
                                ulc_far,
                                urc_far);

  //
  // Ray meshes
  //

  write_blueprint_ray_corners_mesh(data,
                                   llc_near,
                                   llc_far,
                                   lrc_near,
                                   lrc_far,
                                   urc_near,
                                   urc_far,
                                   ulc_near,
                                   ulc_far);

  // This mesh can be very large depending on the image width and height, and it won't
  // always be useful. We only include it in the output if the user explicitly asks for it.
  const bool enable_rays_mesh = rover::settings["enable_rays_mesh"].as_string() == "true";
  if (enable_rays_mesh)
  {
    write_blueprint_rays_mesh(data,
                              image_width,
                              image_height,
                              detector_width,
                              detector_height,
                              lrc_near,
                              far_detector_width,
                              far_detector_height,
                              lrc_far,
                              left,
                              up);
  }

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
