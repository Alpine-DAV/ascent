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

  ~InternalsType()
  {
    if (m_scheduler)
    {
      delete m_scheduler;
    }
  }

  void add_data_set(vtkmDataSet &dataset)
  {
    ROVER_INFO("Adding data set");
    m_scheduler->add_data_set(dataset);
  }

  void set_settings(const Node &settings)
  {
    ROVER_INFO("Executing InternalsType::set_settings");
    // TODO: make copy constructors to get the members like ray_generator
//#ifdef ROVER_PARALLEL
    // logic to create the appropriate parallel scheduler
    //
    // ray tracing = dynamic scheduler, scattering | no_scattering
    // energy = scattering + local_scope -> dynamic scheduler
    //                 non_scattering + global_scope ->static scheduler
    //
    // Note: I wanted to allow for the case of scattering + global scope. This could
    //       be benificial in the case where we may or may not scatter in a given
    //       domain. Thus, avoid waiting for the ray to emerge or throw out the results
//#else
     //if(render_settings compared to old means new schedular)
     //if(m_scheduler == NULL) delete m_scheduler;
     //m_scheduler = new Scheduler<FloatType>();
     m_scheduler->set_settings(settings);
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
  m_settings["rover/color_table"] = "Cool to Warm";
  m_settings["rover/divide_emission_by_abs"] = "false";
  m_settings["rover/num_samples"] = 400;
  m_settings["rover/ray_scope"] = "global_rays";
  m_settings["rover/scattering_type"] = "non_scattering";  
  m_settings["rover/unit_scalar"] = 1.0;
}

Rover::~Rover()
{
#ifdef ROVER_ENABLE_LOGGING
  DataLogger::GetInstance()->WriteLog();
#endif
}

void
Rover::update_settings(Node &params)
{
  if (params.has_child("rover"))
  {
    std::vector<std::string> rover_param_names = params["rover"].child_names();
    for (const auto &param_name : rover_param_names)
    {
      const std::string path = "rover/" + param_name;
      m_settings[path].set(params[path]);
    }
  }

  if (params.has_child("camera"))
  {
    m_settings["camera"].set(params["camera"]);
  }
}

void
Rover::print_settings()
{
  std::cout << m_settings.to_yaml() << std::endl;
}

void
Rover::set_mpi_comm_handle(int mpi_comm_id)
{
#ifdef ROVER_PARALLEL
  this->m_internals->set_comm_handle(MPI_Comm_f2c(mpi_comm_id));
#else
  (void)mpi_comm_id;
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
  m_camera.ResetToBounds(dataset.GetGlobalBounds());
}

void
Rover::update_camera()
{
  // Early return if the default params weren't changed
  if (!m_settings.has_child("camera"))
  {
    return;
  }

  // TODO: Change each instance of has_path to has_child
  if (m_settings.has_path("camera/look_at"))
  {
    float64_accessor vec3 = m_settings["camera/look_at"].value();
    vtkmVec3f look_at(vec3[0], vec3[1], vec3[2]);
    m_camera.SetLookAt(look_at);
  }
  
  if (m_settings.has_path("camera/up"))
  {
    float64_accessor vec3 = m_settings["camera/up"].value();
    vtkmVec3f up(vec3[0], vec3[1], vec3[2]);
    m_camera.SetViewUp(up);
  }
  
  if (m_settings.has_path("camera/fov"))
  {
    float64 fov = m_settings["camera/fov"].value();
    m_camera.SetFieldOfView(fov);
  }
  
  if (m_settings.has_path("camera/xpan") || m_settings.has_path("camera/ypan"))
  {
    float64 xpan = 0.0;
    float64 ypan = 0.0;

    if (m_settings.has_path("camera/xpan"))
    {
      xpan = m_settings["camera/xpan"].value();
    }

    if (m_settings.has_path("camera/ypan"))
    {
      ypan = m_settings["camera/ypan"].value();
    }
    
    m_camera.Pan(xpan, ypan);
  }
  
  if (m_settings.has_path("camera/zoom"))
  {
    float64 image_zoom = m_settings["camera/zoom"].value();
    m_camera.Zoom(log(image_zoom) / log(4.0));
  }
  
  if (m_settings.has_path("camera/near_plane") || m_settings.has_path("camera/far_plane"))
  {
    vtkm::Range clipping_range;

    if (m_settings.has_path("camera/near_plane"))
    {
      clipping_range.Min = m_settings["camera/near_plane"].value();
    }
    
    if (m_settings.has_path("camera/far_plane"))
    {
      clipping_range.Max = m_settings["camera/far_plane"].value();
    }

    m_camera.SetClippingRange(clipping_range);
  }
}

void
Rover::update_ray_generator()
{
  m_ray_generator.set_camera(m_camera);

  int64 width = 200;
  int64 height = 200;

  if (m_settings.has_path("camera/image_width"))
  {
    width = m_settings["camera/image_width"].value();
  }

  if (m_settings.has_path("camera/image_height"))
  {
    height = m_settings["camera/image_height"].value();
  }

  m_ray_generator.set_width(width);
  m_ray_generator.set_height(height);

  m_internals->set_ray_generator(&m_ray_generator);
}

void
Rover::update_precision()
{
  // Precision is an optional parameter
  if (!m_settings.has_path("rover/precision"))
  {
    // Use float32 precision by default
    m_internals->set_tracer_precision32();
  }

  std::string precision = m_settings["rover/precision"].as_string();
  if (precision == "single")
  {
    m_internals->set_tracer_precision32();
  }
  else if (precision == "double")
  {
    m_internals->set_tracer_precision64();
  }
}

void
Rover::clear_data_sets()
{
  m_internals->clear_data_sets();
}

void
Rover::execute()
{
  update_camera();
  update_ray_generator();
  update_precision();
  m_internals->set_settings(m_settings);
  m_internals->execute();
}

template<typename T>
bool
is_float(T);

template<>
bool
is_float<vtkm::Float32>(vtkm::Float32)
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

}; //namespace rover
