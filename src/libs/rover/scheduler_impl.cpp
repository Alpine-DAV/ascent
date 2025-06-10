//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "settings.hpp"
#include <assert.h>
#include <fstream>
#include <vtkh/compositing/PartialCompositor.hpp>
#include <scheduler_impl.hpp>
#include <png_utils/ascent_png_encoder.hpp>
#include <utils/rover_logging.hpp>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm_typedefs.hpp>
#include <ray_generators/camera_generator.hpp>
#include <rover_exceptions.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

using namespace conduit;

namespace rover
{

template<typename FloatType>
SchedulerImpl<FloatType>::SchedulerImpl()
{
  m_ray_generator = nullptr;
}

template<typename FloatType>
SchedulerImpl<FloatType>::~SchedulerImpl()
{

}

template<typename FloatType>
void
SchedulerImpl<FloatType>::add_data_set(vtkmDataSet &dataset)
{
  ROVER_INFO("SchedulerImpl::add_data_set: adding domain " << m_domains.size());
  Domain domain;
  domain.set_data_set(dataset);
  m_domains.push_back(domain);
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_ray_generator(RayGenerator *ray_generator)
{
  m_ray_generator = ray_generator;
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_background(const std::vector<vtkm::Float32> &background)
{
  const size_t size = background.size();
  m_background.resize(size);

  for(size_t i = 0; i < size; ++i)
  {
    m_background[i] = static_cast<vtkm::Float64>(background[i]);
  }
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_background(const std::vector<vtkm::Float64> &background)
{
  m_background = background;
}

template<typename FloatType>
std::vector<Domain>
SchedulerImpl<FloatType>::get_domains()
{
  return m_domains;
}

template<typename FloatType>
vtkmDataSet
SchedulerImpl<FloatType>::get_data_set(const int &domain)
{
  return m_domains.at(domain).get_data_set();
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::create_default_background(const int num_channels)
{
  m_background.resize(num_channels);
  for(int i = 0; i < num_channels; ++i)
  {
    m_background[i] = 0.0f;
  }
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_domains(std::vector<Domain> &domains)
{
  m_domains = domains;
}

#ifdef ROVER_PARALLEL
template<typename FloatType>
void
SchedulerImpl<FloatType>::set_comm_handle(MPI_Comm comm_handle)
{
  m_comm_handle = comm_handle;
}
#endif

template<typename FloatType>
int
SchedulerImpl<FloatType>::get_global_channels()
{
  int num_channels = 1;
  for(size_t i = 0; i < m_domains.size(); ++i)
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

  ROVER_INFO("Global number of channels"<<num_channels);
  return num_channels;
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_global_scalar_range()
{
  vtkmTimer timer;
  timer.Start();
  double time = 0;
  (void) time;

  const int num_domains = static_cast<int>(m_domains.size());

  vtkmRange global_range;

#if 0 // removing volume renderer
  if(m_render_settings.m_render_mode == volume &&
     m_render_settings.m_volume_settings.m_scalar_range.IsNonEmpty())
  {
    global_range = m_render_settings.m_volume_settings.m_scalar_range;
    ROVER_INFO("Provided scalar range "<<global_range);
  }
  else
  {
#endif

  for (int i = 0; i < num_domains; ++i)
  {
    vtkmRange local_range = m_domains[i].get_primary_range();
    global_range.Include(local_range);
  }

#ifdef ROVER_PARALLEL
  double rank_min = global_range.Min;
  double rank_max = global_range.Max;
  double mpi_min;
  double mpi_max;
  MPI_Allreduce(&rank_min, &mpi_min, 1, MPI_DOUBLE, MPI_MIN, m_comm_handle);
  MPI_Allreduce(&rank_max, &mpi_max, 1, MPI_DOUBLE, MPI_MAX, m_comm_handle);
  global_range.Min = mpi_min;
  global_range.Max = mpi_max;
#endif

  ROVER_INFO("Global scalar range: " << global_range);
  // } // removing volume renderer

  for(int i = 0; i < num_domains; ++i)
  {
    m_domains[i].set_primary_range(global_range);
  }

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("set_global_scalar_range", time);
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::set_global_bounds()
{
  vtkmTimer timer;
  timer.Start();
  double time = 0;

  const int num_domains = static_cast<int>(m_domains.size());

  vtkm::Bounds global_bounds;

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Bounds local_bounds = m_domains[i].get_domain_bounds();
    global_bounds.Include(local_bounds);
  }

#ifdef ROVER_PARALLEL
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

  ROVER_INFO("Global bounds "<<global_bounds);

  for(int i = 0; i < num_domains; ++i)
  {
    m_domains[i].set_global_bounds(global_bounds);
  }
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("set_global_bounds", time);
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial)
{
  PartialImage<FloatType> partial_image;
  partial_image.m_pixel_ids = partial.PixelIds;
  partial_image.m_distances = partial.Distances;
  partial_image.m_buffer = partial.Buffer;
  partial_image.m_intensities = partial.Intensities;
  m_partial_images.push_back(partial_image);
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::composite()
{
  int rank = 0;
#ifdef ROVER_PARALLEL
  MPI_Comm_rank(m_comm_handle, &rank);
#endif

#if 0 // removing volume renderer
  if(m_render_settings.m_render_mode == volume)
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
    for(int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::VolumePartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if(rank == 0)
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
    std::vector<std::vector<vtkh::EmissionPartial<FloatType>>> partials;
    partials.resize(num_partials);
    for(int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::EmissionPartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if(rank == 0)
    {
      // data only valid on rank = 0
      p_result.store(result, m_background);
    }

    m_result = p_result;
  }
  // TODO: Figure out if this else block is relevant for energy renders
  else
  {
    vtkh::PartialCompositor<vtkh::AbsorptionPartial<FloatType>> compositor;
    compositor.set_background(m_background);
#ifdef ROVER_PARALLEL
    compositor.set_comm_handle(MPI_Comm_c2f(m_comm_handle));
#endif
    const int num_partials = m_partial_images.size();
    std::vector<std::vector<vtkh::AbsorptionPartial<FloatType>>> partials;
    partials.resize(num_partials);
    for(int i = 0; i < num_partials; ++i)
    {
      m_partial_images[i].extract_partials(partials[i]);
    }
    std::vector<vtkh::AbsorptionPartial<FloatType>> result;
    compositor.composite(partials, result);
    PartialImage<FloatType> p_result;

    if(rank == 0)
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
SchedulerImpl<FloatType>::trace_rays()
{
  ROVER_INFO("tracing_rays");
  vtkmTimer tot_timer;
  vtkmTimer timer;
  tot_timer.Start();
  timer.Start();
  double time = 0;
  (void) time;
  ROVER_DATA_OPEN("schedule_trace");

  if(m_ray_generator == NULL)
  {
    throw RoverException("Error: ray generator must be set before execute is called");
  }

  m_ray_generator->reset();
  // TODO while (m_generator.has_rays())
  ROVER_INFO("Tracing rays");

  //
  // ensure that the render settings are set
  //
  // TODO: make copy constructor so the mesh stuctures are not rebuilt when moving from
  //       volume to energy and vice versa
  const int num_domains = static_cast<int>(m_domains.size());
  ROVER_INFO("SchedulerImpl set render settings for " << num_domains << " domains");
  for(int i = 0; i < num_domains; ++i)
  {
    m_domains[i].init();
  }

  ROVER_INFO("Done SchedulerImpl set render settings for "<<num_domains<<" domains ");
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("setup", time);

  this->set_global_scalar_range();
  this->set_global_bounds();

  vtkmTimer trace_timer;
  trace_timer.Start();
  for(int i = 0; i < num_domains; ++i)
  {
    vtkmTimer domain_timer;
    domain_timer.Start();
    std::stringstream domain_s;
    domain_s<<"trace_domain_"<<i;
    ROVER_DATA_OPEN(domain_s.str());

    vtkmLogger::GetInstance()->Clear();
    if(dynamic_cast<CameraGenerator*>(m_ray_generator) != NULL)
    {
      //
      // Setting the coordinate system miminizes the number of rays generated
      //
      CameraGenerator *generator = dynamic_cast<CameraGenerator*>(m_ray_generator);
      generator->set_coordinates(m_domains[i].get_data_set().GetCoordinateSystem());
    }
    ROVER_INFO("Generating rays for domian "<<i);

    timer.Start();

    vtkmRayTracing::Ray<FloatType> rays;
    m_ray_generator->get_rays(rays);

    ROVER_INFO("Generated "<<rays.NumRays<<" rays");
    m_domains[i].init_rays(rays);
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_init_rays", time);

    ROVER_INFO("Tracing domain "<<i);

    timer.Start();
    std::vector<vtkmRayTracing::PartialComposite<FloatType>> partials;
    partials = m_domains[i].partial_trace(rays);
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_trace", time);
#ifdef ROVER_ENABLE_LOGGING
    DataLogger::GetInstance()->GetStream()<<vtkmLogger::GetInstance()->GetStream().str();
#endif
    ROVER_INFO("Schedule: creating partial image in domain "<<i);
    //
    // Create a partial images from the completed rays
    //
    for(size_t p = 0; p < partials.size(); ++p)
    {
      add_partial(partials[p]);
    }

    timer.Start();
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_push_back", time);

    time = domain_timer.GetElapsedTime();
    ROVER_DATA_CLOSE(time);
    ROVER_INFO("Schedule: done tracing domain "<<i);
  }// for each domain

  timer.Start();
  time = trace_timer.GetElapsedTime();
  ROVER_DATA_ADD("total_trace", time);
  int num_channels = this->get_global_channels();

  vtkmTimer t1;
  t1.Start();

  // Add dummy partial image if we had no domains
  if(num_domains == 0 || m_partial_images.size() == 0)
  {
    PartialImage<FloatType> partial_image;
    partial_image.m_buffer =
      vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);

    const std::string emission = rover::settings["rover/emission"].as_string();
    if("" != emission)
    {
      partial_image.m_intensities =
        vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);
    }
    m_partial_images.push_back(partial_image);
  }
  //DataLogger::GetInstance()->AddLogData("blank_image", t1.GetElapsedTime());
  //ROVER_DATA_ADD("blank_image", t1.GetElapsedTime());
  t1.Start();

  if(m_background.size() == 0)
  {
    this->create_default_background(num_channels);
  }

  ROVER_DATA_ADD("default_bg", t1.GetElapsedTime());
  t1.Start();

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("mid", t1.GetElapsedTime());
  timer.Start();

  //
  // Composite the results
  //
  timer.Start();
  composite();
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("compositing", time);
  timer.Start();

  m_partial_images.clear();
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("clear", time);

  double tot_time = tot_timer.GetElapsedTime();
  (void) tot_time;
  ROVER_DATA_CLOSE(tot_time);
  ROVER_INFO("Schedule: end of trace");
}

template<typename FloatType>
void SchedulerImpl<FloatType>::save_result(std::string file_name)
{
  const int32 width = rover::settings["rover/width"].value();
  const int32 height = rover::settings["rover/height"].value();
  ROVER_INFO("Saving .png file with output size " << width << "x" << height);
  ascent::PNGEncoder encoder;

  // if(m_render_settings.m_render_mode == energy) // removing volume renderer
  // {

  const int num_channels = m_result.get_num_channels();
  ROVER_INFO("Saving "<<num_channels<<" channels ");
  for(int i = 0; i < num_channels; ++i)
  {
    std::stringstream sstream;
    sstream<<file_name<<"_"<<i<<".png";
    m_result.normalize_intensity(i);
    FloatType * buffer
      = get_vtkm_ptr(m_result.get_intensity(i));

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
void
SchedulerImpl<FloatType>::save_result(std::string file_name,
                                  float min_val,
                                  float max_val,
                                  bool log_scale)
{
  const int32 width = rover::settings["rover/width"].value();
  const int32 height = rover::settings["rover/height"].value();
  ROVER_INFO("Saving .png file with output size " << width << "x" << height);
  ascent::PNGEncoder encoder;

#if 0 // removing volume renderer
  if(!(m_render_settings.m_render_mode == energy))
  {
    throw RoverException("Error: can only save images with min and max in energy mode");
  }
#endif

  const int num_channels = m_result.get_num_channels();
  ROVER_INFO("Saving " << num_channels << " channels");
  for(int i = 0; i < num_channels; ++i)
  {
    std::stringstream sstream;
    sstream<<file_name<<"_"<<i<<".png";
    m_result.normalize_intensity(i, min_val, max_val, log_scale);
    FloatType * buffer
      = get_vtkm_ptr(m_result.get_intensity(i));

    encoder.EncodeChannel(buffer, width, height);
    encoder.Save(sstream.str());
  }
}

template<typename FloatType>
void SchedulerImpl<FloatType>::save_bov(std::string file_name)
{
  const int32 width = rover::settings["rover/width"].value();
  const int32 height = rover::settings["rover/height"].value();
  ROVER_INFO("Saving bov file with output size " << width << "x" << height);

  ascent::PNGEncoder encoder;
  const int size = height * width;

  // if(m_render_settings.m_render_mode == energy) // removing volume renderer
  // {
    
  const int num_channels = m_result.get_num_channels();
  ROVER_INFO("Saving bov"<<num_channels<<" channels ");
  for(int i = 0; i < num_channels; ++i)
  {
    std::stringstream sstream;
    sstream<<file_name<<"_"<<i<<".bov";
    m_result.normalize_intensity(i);
    FloatType * buffer
      = get_vtkm_ptr(m_result.get_intensity(i));
    std::fstream bov(sstream.str(), std::ios::out | std::ios::binary);
    bov.write((char*)buffer, sizeof(FloatType) * size);
    bov.close();
  }
  // } // removing volume renderer
}

template<typename FloatType>
void
SchedulerImpl<FloatType>::to_blueprint(Node &data)
{
  const int32 width = rover::settings["rover/width"].value();
  const int32 height = rover::settings["rover/height"].value();
  ROVER_INFO("Saving blueprint file with output size " << width << "x" << height);

  // TODO: Plumb the other "state/" info down out of *_rover_filters.cpp
  Node &xray_view = data["state/xray_view"];
  vtkmCamera camera = m_ray_generator->get_camera();
  xray_view["position"].set(&camera.GetPosition()[0], 3);
  xray_view["look_at"].set(&camera.GetLookAt()[0], 3);
  xray_view["up"].set(&camera.GetViewUp()[0], 3);
  xray_view["zoom"] = camera.GetZoom();
  xray_view["fov"] = camera.GetFieldOfView();
  xray_view["near_plane"] = camera.GetClippingRange().Min;
  xray_view["far_plane"] = camera.GetClippingRange().Max;

  auto xy_pan = camera.GetPan();
  xray_view["xpan"] = xy_pan[0];
  xray_view["ypan"] = xy_pan[1];

  Node &xray_query = data["xray_query"];
  xray_query.set(rover::settings["rover"]);

  const int num_channels = m_result.get_num_channels();
  const std::string coord_name = "image_coords";
  Node &n_coords = data["coordsets"][coord_name];
  n_coords["type"] = "rectilinear";
  
  n_coords["values/x"].set(DataType::float32(width + 1));
  n_coords["values/y"].set(DataType::float32(height + 1));
  n_coords["values/z"].set(DataType::float32(num_channels + 1));

  float32_array x_coords = n_coords["values/x"].value();
  float32_array y_coords = n_coords["values/y"].value();
  float32_array z_coords = n_coords["values/z"].value();

  for (int i = 0; i <= width; i++)
  {
    x_coords[i] = i;
  }

  for (int i = 0; i <= height; i++)
  {
    y_coords[i] = i;
  }

  for (int i = 0; i <= num_channels; i++)
  {
    z_coords[i] = i;
  }

  n_coords["labels/x"] = "width";
  n_coords["labels/y"] = "height";
  n_coords["labels/z"] = "energy_group";

  n_coords["units/x"] = "pixels";
  n_coords["units/y"] = "pixels";
  n_coords["units/z"] = "bins";

  const std::string topo_name = "image_topo";
  Node &n_topo = data["topologies"][topo_name];
  n_topo["coordset"] = coord_name;
  n_topo["type"] = "rectilinear";

  // if(m_render_settings.m_render_mode == energy) // removing volume renderer
  // {

  if (!m_result.has_intensity(0) || !m_result.has_optical_depth(0))
  {
    ROVER_ERROR("intensity and optical depth must both be available")
  }

  // Intensity
  Node &intensities = data["fields/intensities"];
  intensities["topology"] = topo_name;
  intensities["association"] = "element";
  intensities["units"] = "intensity units";
  vtkm::cont::ArrayHandle<FloatType> intensity_values = m_result.flatten_intensities();
  FloatType *intensity_buffer = get_vtkm_ptr(intensity_values);
  // can't set external since this goes out of scope
  const int num_intensity_values = intensity_values.GetNumberOfValues();
  intensities["values"].set(intensity_buffer, num_intensity_values);
  intensities["strides"].set(DataType::int64(3));
  int64_array strides = intensities["strides"].value();
  strides[0] = 1;
  strides[1] = width;
  strides[2] = width * height;

  // Optical Depth
  Node &optical_depth = data["fields/optical_depth"];
  optical_depth["topology"] = topo_name;
  optical_depth["association"] = "element";
  optical_depth["units"] = "path length metadata";
  vtkm::cont::ArrayHandle<FloatType> optical_values = m_result.flatten_optical_depths();
  FloatType *optical_buffer = get_vtkm_ptr(optical_values);
  // can't set external since this goes out of scope
  const int num_optical_values = optical_values.GetNumberOfValues();
  optical_depth["values"].set(optical_buffer, num_optical_values);
  optical_depth["strides"].set(intensities["strides"]);

  // } // removing volume renderer

  Node verify;
  if(!blueprint::verify("mesh", data, verify))
  {
    ROVER_ERROR("Error: to_blueprint failed to produce a valid conduit mesh: " << verify.to_yaml());
  }
}

//
// Explicit instantiation
template class SchedulerImpl<vtkm::Float32>;
template class SchedulerImpl<vtkm::Float64>;
}; // namespace rover
