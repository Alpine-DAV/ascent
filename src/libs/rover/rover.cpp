//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "image.hpp"
#include <scheduler.hpp>
#include <rover.hpp>
#include <rover_exceptions.hpp>
#include <vtkm_typedefs.hpp>
#include <iostream>
#include <utils/rover_logging.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

namespace rover {

class Rover::InternalsType
{
public:
  enum TracePrecision {ROVER_FLOAT, ROVER_DOUBLE};
protected:
  SchedulerBase            *m_scheduler;
  TracePrecision            m_precision;
#ifdef ROVER_PARALLEL
  MPI_Comm                  m_comm_handle;
  int                       m_rank;
  int                       m_num_ranks;
#endif

  void reset_render_mode(RenderMode render_mode)
  {

  }

public:
  InternalsType()
  {
    m_precision = ROVER_FLOAT;
    m_scheduler = new Scheduler<vtkm::Float32>();

#ifdef ROVER_PARALLEL
    m_rank = 1;
    m_num_ranks = -1;
#endif
  }

  void add_data_set(vtkmDataSet &dataset)
  {
    ROVER_INFO("Adding data set");
    m_scheduler->add_data_set(dataset);
  }

  void set_render_settings(RenderSettings render_settings)
  {
    ROVER_INFO("set_render_settings");
    // TODO: make copy constructors to get the members like ray_generator
//#ifdef ROVER_PARALLEL
    // logic to create the appropriate parallel scheduler
    //
    // ray tracing = dynamic scheduler, scattering | no_scattering
    // volume/engery = scattering + local_scope -> dynamic scheduler
    //                 non_scattering + global_scope ->static scheduler
    //
    // Note: I wanted to allow for the case of scattering + global scope. This could
    //       be benificial in the case where we may or may not scatter in a given
    //       domain. Thus, avoid waiting for the ray to emerge or throw out the results
//#else
     //if(render_settings compared to old means new schedular)
     //if(m_scheduler == NULL) delete m_scheduler;
     //m_scheduler = new Scheduler<FloatType>();
     m_scheduler->set_render_settings(render_settings);
//#endif
  }

  void set_ray_generator(RayGenerator *ray_generator)
  {
    m_scheduler->set_ray_generator(ray_generator);
  }

  void clear_data_sets()
  {
    m_scheduler->clear_data_sets();
  }

  ~InternalsType()
  {
    if(m_scheduler) delete m_scheduler;
  }

  void set_background(const std::vector<vtkm::Float32> &background)
  {
    m_scheduler->set_background(background);
  }

  void set_background(const std::vector<vtkm::Float64> &background)
  {
    m_scheduler->set_background(background);
  }

  void to_blueprint(conduit::Node &dataset)
  {
#ifdef ROVER_PARALLEL
    if(m_rank != 0)
    {
      return;
    }
#endif
    m_scheduler->to_blueprint(dataset);
  }

  void save_png(const std::string &file_name)
  {
#ifdef ROVER_PARALLEL
    if(m_rank != 0)
    {
      return;
    }
#endif
    m_scheduler->save_result(file_name);
  }
  void save_png(const std::string &file_name,
                const float min_val,
                const float max_val,
                const bool log_scale)
  {
#ifdef ROVER_PARALLEL
    if(m_rank != 0)
    {
      return;
    }
#endif
    m_scheduler->save_result(file_name, min_val, max_val, log_scale);
  }

  void save_bov(const std::string &file_name)
  {
#ifdef ROVER_PARALLEL
    if(m_rank != 0)
    {
      return;
    }
#endif
    m_scheduler->save_bov(file_name);
  }

  void execute()
  {
#ifdef ROVER_PARALLEL
    //
    // Check to see if we have been initialized
    //
    if(m_rank == -1)
    {
      ROVER_ERROR("Execute call with MPI enbaled, but never initialized with comm handle");
    }

    m_scheduler->set_comm_handle(m_comm_handle);
#endif
    m_scheduler->trace_rays();
  }
#ifdef ROVER_PARALLEL
  void set_comm_handle(MPI_Comm comm_handle)
  {
    m_comm_handle = comm_handle;
    MPI_Comm_rank(m_comm_handle, &m_rank);
    if(m_rank == 0)
    {
      MPI_Comm_size(m_comm_handle, &m_num_ranks);
      ROVER_INFO("MPI Comm size : "<<m_num_ranks);
    }
  }

  MPI_Comm get_comm_handle()
  {
    return m_comm_handle;
  }
#endif
  void get_result(Image<vtkm::Float32> &image)
  {
    m_scheduler->get_result(image);
  }

  void get_result(Image<vtkm::Float64> &image)
  {
    m_scheduler->get_result(image);
  }

  void set_tracer_precision32()
  {
    if(m_precision == ROVER_DOUBLE)
    {
      std::vector<Domain> domains = m_scheduler->get_domains();
      delete m_scheduler;
      m_scheduler = new Scheduler<vtkm::Float32>();
    }
  }

  void set_tracer_precision64()
  {
    if(m_precision == ROVER_FLOAT)
    {
      std::vector<Domain> domains = m_scheduler->get_domains();
      delete m_scheduler;
      m_scheduler = new Scheduler<vtkm::Float64>();
    }
  }

}; //Internals Type

Rover::Rover() : m_internals( new InternalsType )
{
  // Rover settings
  render_mode = energy;
  scattering_type = non_scattering;
  ray_scope = global_rays;
  color_table.LoadPreset("Cool to Warm");

  // Energy settings
  divide_abs_by_emission = false;
  unit_scalar = 1.0;

  // Volume setings
  num_samples = 400;

  // Camera settings
  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 1.0;
  focus[0] = 0.0;
  focus[1] = 0.0;
  focus[2] = 0.0;
  viewUp[0] = 0.0;
  viewUp[1] = 1.0;
  viewUp[2] = 0.0;
  viewAngle = 30.0;
  parallelScale = 0.5;
  viewWidthOverride = 0.0;
  nonSquarePixels = false;
  nearPlane = -0.5;
  farPlane = 0.5;
  imagePan[0] = 0.0;
  imagePan[1] = 0.0;
  imageZoom = 1.0;
  perspective = true;
  imageSize[0] = 200;
  imageSize[1] = 200;

  Rover::initialize_camera();
  // Rover::initialize_camera_generator();
}

Rover::~Rover()
{
#ifdef ROVER_ENABLE_LOGGING
  DataLogger::GetInstance()->WriteLog();
#endif
}

void
Rover::initialize_camera()
{
  vtkmVec3f look_at(focus[0], focus[1], focus[2]);
  vtkmVec3f up(viewUp[0], viewUp[1], viewUp[2]);

  camera.SetLookAt(look_at);
  camera.SetViewUp(up);
  camera.SetFieldOfView(viewAngle);
  camera.Pan(imagePan[0], imagePan[1]);
  camera.Zoom(log(imageZoom) / log(4.0));

  vtkm::Range clipping_range;
  clipping_range.Min = imagePan[0];
  clipping_range.Max = imagePan[1];
  camera.SetClippingRange(clipping_range);
}

void
Rover::initialize_camera_generator()
{
  camera_generator.set_camera(camera);
  camera_generator.set_width(imageSize[0]);
  camera_generator.set_height(imageSize[1]);
  Rover::set_ray_generator(&camera_generator);
}

vtkmCamera&
Rover::get_camera()
{
  return camera;
}

void
Rover::set_mpi_comm_handle(int mpi_comm_id)
{
#ifdef ROVER_PARALLEL
  this->m_internals->set_comm_handle(MPI_Comm_f2c(mpi_comm_id));
#else
  (void) mpi_comm_id;
#endif
}

int
Rover::get_mpi_comm_handle()
{
#ifdef ROVER_PARALLEL
  return MPI_Comm_c2f(this->m_internals->get_comm_handle());
#else
  return -1;
#endif
}

void
Rover::add_data_set(vtkh::DataSet &dataset)
{
  for (int i = 0; i < dataset.GetNumberOfDomains(); i++)
  {
    m_internals->add_data_set(dataset.GetDomain(i));
  }
  // camera.ResetToBounds(dataset.GetGlobalBounds());
}

void
Rover::set_render_settings(RenderSettings render_settings)
{
  m_internals->set_render_settings(render_settings);
}

void
Rover::clear_data_sets()
{
  m_internals->clear_data_sets();
}

void
Rover::set_ray_generator(RayGenerator *ray_generator)
{
  if(ray_generator == nullptr)
  {
    throw RoverException("Ray generator cannot  be null");
  }
  m_internals->set_ray_generator(ray_generator);
}

void
Rover::set_image_dims(int width, int height)
{
  imageSize[0] = width;
  imageSize[1] = height;
  camera_generator.set_image_dims(imageSize[0], imageSize[1]);
}

void
Rover::execute()
{
  m_internals->execute();
}

template<typename T>
bool
is_float(T );

template<>
bool
is_float<vtkm::Float32>(vtkm::Float32 )
{
  return true;
}

template<>
bool
is_float<vtkm::Float64>(vtkm::Float64)
{
  return false;
}

void
Rover::about()
{
  std::cout<<"rover version: xx.xx.xx\n";

  //if(is_float(FloatType())) std::cout<<"Single precision\n";
  //else std::cout<<"Double precision\n";
  std::cout<<"Other important information\n";
  std::cout<<"                                 *@@                                    \n";
  std::cout<<"       @@@@@@@@@@@@@@,          @@&@@              %@@@                 \n";
  std::cout<<"       @@@@@%  #@@@@@,         &@* @@,              @@@#                \n";
  std::cout<<"       @@@@ @    @@@@,         @@   @@             @@ .@@               \n";
  std::cout<<"       @@@@@    #@@@@,        .@&    @@           @@(   @@@             \n";
  std::cout<<"       @@@@@@@@@@@@@@,         @@    %@&         .@@      @@            \n";
  std::cout<<"       &&&&@@&&@@&&&&.     @@@@@@%    @@         @@        @@@          \n";
  std::cout<<"           %@, @@          @@   &@@    @@       @@           @@.        \n";
  std::cout<<"           %@, @@          @@     @@@  ,@@     *@&            (@@       \n";
  std::cout<<"           %@, @@          @@       ,@@@@@     @@               @@(     \n";
  std::cout<<"          *&@(*@@*         @@                 @@                 (@@@#  \n";
  std::cout<<"          @@@@@@@@         @@                @@%                  @@.@@ \n";
  std::cout<<"          @@    @@         @@                @@                   @@  @@\n";
  std::cout<<"          @@    @@        @@@@              @@                    @@ @@.\n";
  std::cout<<"          @@    @@        @@@@             @@                     @@@%  \n";
  std::cout<<"          @@    @@        @@@@         /@@@@@@@@@@@&                    \n";
  std::cout<<"   ,,,,,,,@@,,,,@@,,,,,,,,@@@@,,,,,,,,,(@#,,,,,,,,@@@,,,,,,,,,,,,,,.    \n";
  std::cout<<"  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    \n";
  std::cout<<"  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    \n";
  std::cout<<"     /@@*                                                     @@%       \n";
  std::cout<<"       .@@*                                                .@@%         \n";
  std::cout<<"         *@@.                                             @@#           \n";
  std::cout<<"           ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@(             \n";
  std::cout<<"           #@(                                 @@                       \n";
  std::cout<<"           @@                     /@@@@@@@@@@@@@@@@@@@@@@@@@@@          \n";
  std::cout<<"          @@                      /@/                       @@          \n";
  std::cout<<"      @@@@@@@@(                &@@@@@@@&                #@@@@@@@@       \n";
  std::cout<<"   &@@@@@@@@@@@@@           /@@@@@@@@@@@@@            @@@@@@@@@@@@@%    \n";
  std::cout<<"  @@@@@   @  #@@@@#        @@@@@   @  ,@@@@@        %@@@@(  @   @@@@@   \n";
  std::cout<<" @@@@,@/,@@@ @%@@@@,      %@@@%@( @@@ %@@@@@%      ,@@@@%@ @@@,/@#@@@@  \n";
  std::cout<<" @@@@  .@& @@  *@@@@      @@@@  .@@ @@   @@@@      @@@@   @@ &@.  @@@@  \n";
  std::cout<<" @@@@ #@@@@@@@.@@@@(      @@@@.*@@@@@@@*(@@@@      (@@@# @@@@@@@#.@@@@  \n";
  std::cout<<"  @@@@*   @   @@@@@        @@@@&   @   &@@@@        @@@@@   @   /@@@@   \n";
  std::cout<<"   @@@@@@@@@@@@@@#          @@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@    \n";
  std::cout<<"     @@@@@@@@@@*              &@@@@@@@@@#              #@@@@@@@@@@      \n";

}

void
Rover::set_background(const std::vector<vtkm::Float64> &background)
{
  m_internals->set_background(background);
}

void
Rover::set_background(const std::vector<vtkm::Float32> &background)
{
  m_internals->set_background(background);
}

void Rover::to_blueprint(conduit::Node &dataset)
{
  m_internals->to_blueprint(dataset);
}

void
Rover::save_png(const std::string &file_name)
{
  m_internals->save_png(file_name);
}

void
Rover::save_png(const std::string &file_name,
                const float min_val,
                const float max_val,
                const bool log_scale)
{
  m_internals->save_png(file_name, min_val, max_val, log_scale);
}

void
Rover::save_bov(const std::string &file_name)
{
  m_internals->save_bov(file_name);
}

void
Rover::get_result(Image<vtkm::Float32> &image)
{
  m_internals->get_result(image);
}

void
Rover::get_result(Image<vtkm::Float64> &image)
{
  m_internals->get_result(image);
}

void
Rover::set_tracer_precision32()
{
  m_internals->set_tracer_precision32();
}

void
Rover::set_tracer_precision64()
{
  m_internals->set_tracer_precision64();
}

}; //namespace rover

