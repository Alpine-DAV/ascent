//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-749865
//
// All rights reserved.
//
// This file is part of Rover.
//
// Please also read rover/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <assert.h>
#include <compositing/compositor.hpp>
#include <scheduler.hpp>
#include <utils/png_encoder.hpp>
#include <utils/rover_logging.hpp>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm_typedefs.hpp>
#include <ray_generators/camera_generator.hpp>
#include <rover_exceptions.hpp>


namespace rover {

template<typename FloatType>
Scheduler<FloatType>::Scheduler()
{
  m_ray_generator = NULL;
}

template<typename FloatType>
Scheduler<FloatType>::~Scheduler()
{
}

template<typename FloatType>
int
Scheduler<FloatType>::get_global_channels()
{

  int num_channels = 1;
  for(size_t i = 0; i < m_domains.size(); ++i)
  {
    num_channels = std::max(num_channels, m_domains[i].get_num_channels());
  }
#ifdef PARALLEL
  vtkmTimer timer;
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
Scheduler<FloatType>::set_global_scalar_range()
{

  vtkmTimer timer;
  double time = 0;
  (void) time;

  const int num_domains = static_cast<int>(m_domains.size());

  vtkmRange global_range;

  for(int i = 0; i < num_domains; ++i) 
  {
    vtkmRange local_range = m_domains[i].get_primary_range();
    global_range.Include(local_range);
  }
#ifdef PARALLEL
  double rank_min = global_range.Min;
  double rank_max = global_range.Max;
  double mpi_min;
  double mpi_max;
  MPI_Allreduce(&rank_min, &mpi_min, 1, MPI_DOUBLE, MPI_MIN, m_comm_handle);
  MPI_Allreduce(&rank_max, &mpi_max, 1, MPI_DOUBLE, MPI_MAX, m_comm_handle);
  global_range.Min = mpi_min;
  global_range.Max = mpi_max;
#endif

  ROVER_INFO("Global scalar range "<<global_range);

  for(int i = 0; i < num_domains; ++i) 
  {
    m_domains[i].set_primary_range(global_range);
  }

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("set_global_scalar_range", time);
}

template<typename FloatType>
void
Scheduler<FloatType>::set_global_bounds()
{
  vtkmTimer timer;
  double time = 0;

  const int num_domains = static_cast<int>(m_domains.size());

  vtkm::Bounds global_bounds;

  for(int i = 0; i < num_domains; ++i) 
  {
    vtkm::Bounds local_bounds = m_domains[i].get_domain_bounds();
    global_bounds.Include(local_bounds);
  }

#ifdef PARALLEL

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
void Scheduler<FloatType>::add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial,
                                       int width,
                                       int height)
{
  PartialImage<FloatType> partial_image;
  partial_image.m_pixel_ids = partial.PixelIds;
  partial_image.m_distances = partial.Distances;
  partial_image.m_buffer = partial.Buffer;
  partial_image.m_intensities = partial.Intensities;
  partial_image.m_path_lengths = partial.PathLengths;

  partial_image.m_width = width;
  partial_image.m_height = height;

  m_partial_images.push_back(partial_image);
}

template<typename FloatType>
void Scheduler<FloatType>::composite()
{
  if(m_render_settings.m_render_mode == volume)
  {
    Compositor<VolumePartial<FloatType>> compositor;
    compositor.set_background(m_background);
#ifdef PARALLEL
    compositor.set_comm_handle(m_comm_handle);
#endif
    m_result = compositor.composite(m_partial_images);
  }
  else
  {
    if(m_render_settings.m_secondary_field != "")
    {
      Compositor<EmissionPartial<FloatType>> compositor;
      compositor.set_background(m_background);
#ifdef PARALLEL
      compositor.set_comm_handle(m_comm_handle);
#endif
      m_result = compositor.composite(m_partial_images);
    }
    else
    {
      Compositor<AbsorptionPartial<FloatType>> compositor;
      compositor.set_background(m_background);
#ifdef PARALLEL
        compositor.set_comm_handle(m_comm_handle);
#endif
      m_result = compositor.composite(m_partial_images);
    }
  }
  ROVER_INFO("Schedule: compositing complete");
}
//
// in the other schedulers this method will be far from trivial
//
template<typename FloatType>
void
Scheduler<FloatType>::trace_rays()
{
  ROVER_INFO("tracing_rays");
  vtkmTimer tot_timer;
  vtkmTimer timer;
  double time = 0;
  (void) time;
  ROVER_DATA_OPEN("schedule_trace");

  if(m_ray_generator == NULL)
  {
    throw RoverException("Error: ray generator must be set before execute is called");
  }

  m_ray_generator->reset();
  // TODO while (m_geerator.has_rays())
  ROVER_INFO("Tracing rays");

  int height = 0 ;
  int width = 0;

  m_ray_generator->get_dims(height, width);
  
  //
  // ensure that the render settings are set
  //
  // TODO: make copy constructor so the mesh stuctures are not rebuilt when moving from
  //       volume to energy and vice versa
  const int num_domains = static_cast<int>(m_domains.size());
  ROVER_INFO("scheduer set render settings for "<<num_domains<<" domains ");
  for(int i = 0; i < num_domains; ++i) 
  {
    m_domains[i].set_render_settings(m_render_settings);
  }
  
  ROVER_INFO("done scheduer set render settings for "<<num_domains<<" domains ");
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("setup", time);

  this->set_global_scalar_range();
  this->set_global_bounds();

  vtkmTimer trace_timer;
  for(int i = 0; i < num_domains; ++i)
  {
    vtkmTimer domain_timer;
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

    timer.Reset();
  
    vtkmRayTracing::Ray<FloatType> rays;
    m_ray_generator->get_rays(rays);

    ROVER_INFO("Generated "<<rays.NumRays<<" rays");
    m_domains[i].init_rays(rays);
    //
    // add path lengths if they were requested
    //
    if(m_render_settings.m_path_lengths)
    {
      rays.AddBuffer(1, "path_lengths");
      rays.GetBuffer("path_lengths").InitConst(0);
    }
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_init_rays", time);

    ROVER_INFO("Tracing domain "<<i);

    timer.Reset();
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
      add_partial(partials[p], width, height);
    }

    timer.Reset();
    time = timer.GetElapsedTime();
    ROVER_DATA_ADD("domain_push_back", time);

    time = domain_timer.GetElapsedTime();
    ROVER_DATA_CLOSE(time);
    ROVER_INFO("Schedule: done tracing domain "<<i);
  }// for each domain

  timer.Reset();
  time = trace_timer.GetElapsedTime();
  ROVER_DATA_ADD("total_trace", time);
  int num_channels = this->get_global_channels();

  vtkmTimer t1;

  // Add dummy partial image if we had no domains

  if(num_domains == 0 || m_partial_images.size() == 0)
  {
    PartialImage<FloatType> partial_image;
    partial_image.m_width = width;
    partial_image.m_height = height;
    partial_image.m_buffer =
      vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);
    if(m_render_settings.m_secondary_field != "")
    {
      partial_image.m_intensities =
        vtkm::rendering::raytracing::ChannelBuffer<FloatType>(num_channels, 0);
    }
    m_partial_images.push_back(partial_image);
  }
  //DataLogger::GetInstance()->AddLogData("blank_image", t1.GetElapsedTime());
  //ROVER_DATA_ADD("blank_image", t1.GetElapsedTime());
  t1.Reset();

  if(m_background.size() == 0)
  {
    this->create_default_background(num_channels);
  }

  ROVER_DATA_ADD("default_bg", t1.GetElapsedTime());
  t1.Reset();

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("mid", t1.GetElapsedTime());
  timer.Reset();
  
  // 
  // Composite the results
  // 
  timer.Reset();
  composite(); 
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("compositing", time);
  timer.Reset();

  m_partial_images.clear();
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("clear", time);

  double tot_time = tot_timer.GetElapsedTime();
  (void) tot_time;
  ROVER_DATA_CLOSE(tot_time);
  ROVER_INFO("Schedule: end of trace");
}

template<typename FloatType>
void
Scheduler<FloatType>::get_result(Image<vtkm::Float32> &image)
{
  image = m_result;
}

template<typename FloatType>
void
Scheduler<FloatType>::get_result(Image<vtkm::Float64> &image)
{
  image = m_result;
}

template<typename FloatType>
void Scheduler<FloatType>::save_result(std::string file_name) 
{
  int height = 0;
  int width = 0;
  m_ray_generator->get_dims(height, width);
  assert( height > 0 );
  assert( width > 0 );
  ROVER_INFO("Saving file " << height << " "<<width);
  PNGEncoder encoder;

  if(m_render_settings.m_render_mode == energy)
  {
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

  if(m_render_settings.m_path_lengths)
  {
     std::stringstream sstream;
     sstream<<file_name<<"_paths"<<".png";
     m_result.normalize_paths();
     FloatType * buffer 
       = get_vtkm_ptr(m_result.get_path_lengths());

     encoder.EncodeChannel(buffer, width, height);
     encoder.Save(sstream.str());
  }
  
}


//
// Explicit instantiation
template class Scheduler<vtkm::Float32>; 
template class Scheduler<vtkm::Float64>;
}; // namespace rover
