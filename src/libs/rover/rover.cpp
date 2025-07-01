//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <typed_scheduler.hpp>
#include <rover.hpp>
#include <rover_exceptions.hpp>
#include <vtkm_typedefs.hpp>
#include <iostream>
#include <utils/rover_logging.hpp>
#include <settings.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

namespace rover
{

// Namespaced settings instance
Node metadata;
Node settings;

Rover::Rover()
{
  // Ensure that we always start from a default state
  rover::metadata.reset();
  rover::settings.reset();

  // Settings
  // TODO: Figure out if color_table needs to be set by us here
  rover::settings["color_table"] = "Cool to Warm";
  rover::settings["divide_emis_by_absorb"] = "false";
  rover::settings["height"] = 200;
  rover::settings["precision"] = "single";
  rover::settings["ray_scope"] = "global_rays";
  rover::settings["scattering_type"] = "non_scattering";
  rover::settings["width"] = 200;
  rover::settings["unit_scalar"] = 1.0f;

  // We don't instantiate the scheduler until we determine if the
  // user has requested a specific precision to use
  m_scheduler = nullptr;

#ifdef ROVER_PARALLEL
  m_rank = 1;
  m_num_ranks = -1;
#endif
}

Rover::~Rover()
{
  if (m_scheduler)
  {
    delete m_scheduler;
  }

#ifdef ROVER_ENABLE_LOGGING
  DataLogger::GetInstance()->WriteLog();
#endif
}

void
Rover::update_time_and_cycle(const Node &metadata)
{
  if (metadata.has_child("time"))
  {
    rover::metadata["time"].set(metadata["time"]);
  }

  if (metadata.has_child("cycle"))
  {
    rover::metadata["cycle"].set(metadata["cycle"]);
  }
}

void
Rover::update_settings(const Node &params)
{
  ROVER_INFO("Executing Rover::update_settings");

  if (params.has_child("rover"))
  {
    rover::settings.update(params["rover"]);
  }

  if (params.has_child("camera"))
  {
    rover::settings["camera"].set(params["camera"]);
  }
}

void
Rover::print_settings()
{
  std::cout << rover::settings.to_yaml() << std::endl;
}

// TODO: Validate correctness
#ifdef ROVER_PARALLEL
void
Rover::set_mpi_comm_handle(int comm_handle)
{
  m_comm_handle = MPI_Comm_f2c(comm_handle);
  MPI_Comm_rank(m_comm_handle, &m_rank);
  if (0 == m_rank)
  {
    MPI_Comm_size(m_comm_handle, &m_num_ranks);
    ROVER_INFO("MPI Comm size: " << m_num_ranks);
  }
}

int
Rover::get_mpi_comm_handle()
{
  return MPI_Comm_c2f(m_comm_handle);
}
#endif

void
Rover::create_scheduler()
{
  const std::string precision = rover::settings["precision"].as_string();
  if ("double" == precision)
  {
    m_scheduler = new TypedScheduler<vtkm::Float64>();
  }
  else // ("single" == precision)
  {
    m_scheduler = new TypedScheduler<vtkm::Float32>();
  }

#ifdef ROVER_PARALLEL
  // Check to see if we have been initialized
  if(-1 == m_rank)
  {
    ROVER_ERROR("Error - Rover::create_scheduler: MPI was not initialized");
  }
  m_scheduler->set_comm_handle(m_comm_handle);
#endif
}

void
Rover::add_dataset(vtkh::DataSet &dataset)
{
  ROVER_INFO("Executing Rover::add_dataset");
  // The scheduler needs to be created before data can be added
  // to it, else we segfault
  if (!m_scheduler)
  {
    create_scheduler();
  }

  for (int i = 0; i < dataset.GetNumberOfDomains(); i++)
  {
    m_scheduler->add_dataset(dataset.GetDomain(i));
  }

  m_camera.ResetToBounds(dataset.GetGlobalBounds());
}

void
Rover::update_camera()
{
  // Early return if the default params weren't changed
  if (!rover::settings.has_child("camera"))
  {
    return;
  }

  // The order in which these parameters are applied matters
  const Node &camera_params = rover::settings["camera"];

  if (camera_params.has_child("azimuth"))
  {
    const float64 azimuth = camera_params["azimuth"].to_float64();
    m_camera.Azimuth(azimuth);
  }

  if (camera_params.has_child("elevation"))
  {
    const float64 elevation = camera_params["elevation"].to_float64();
    m_camera.Elevation(elevation);
  }

  if (camera_params.has_child("zoom"))
  {
    const float64 zoom = camera_params["zoom"].to_float64();
    m_camera.Zoom(log(zoom) / log(4.0));
  }

  if (camera_params.has_child("look_at"))
  {
    const float64_accessor vec3 = camera_params["look_at"].value();
    const vtkmVec3f look_at(vec3[0], vec3[1], vec3[2]);
    m_camera.SetLookAt(look_at);
  }
  
  if (camera_params.has_child("up"))
  {
    const float64_accessor vec3 = camera_params["up"].value();
    const vtkmVec3f up(vec3[0], vec3[1], vec3[2]);
    m_camera.SetViewUp(up);
  }
  
  if (camera_params.has_child("fov"))
  {
    const float64 fov = camera_params["fov"].to_float64();
    m_camera.SetFieldOfView(fov);
  }
  
  const bool has_xpan = camera_params.has_child("xpan");
  const bool has_ypan = camera_params.has_child("ypan");

  if (has_xpan || has_ypan)
  {
    const vtkmVec2f pan = m_camera.GetPan();
    float64 xpan = pan[0];
    float64 ypan = pan[1];

    if (has_xpan)
    {
      xpan = camera_params["xpan"].to_float64();
    }

    if (has_ypan)
    {
      ypan = camera_params["ypan"].to_float64();
    }
    
    m_camera.Pan(xpan, ypan);
  }
  
  const bool has_near_plane = camera_params.has_child("near_plane");
  const bool has_far_plane = camera_params.has_child("far_plane");

  if (has_near_plane || has_far_plane)
  {
    vtkm::Range clipping_range = m_camera.GetClippingRange();

    if (has_near_plane)
    {
      clipping_range.Min = camera_params["near_plane"].to_float64();
    }
    
    if (has_far_plane)
    {
      clipping_range.Max = camera_params["far_plane"].to_float64();
    }

    m_camera.SetClippingRange(clipping_range);
  }
}

void
Rover::update_ray_generator()
{
  m_ray_generator.set_camera(m_camera);
  m_scheduler->set_ray_generator(&m_ray_generator);
}

void
Rover::execute()
{
  // TODO: Not sure if this needs to be a full error. We're not in
  // an unrecoverable state, we just simply have nothing to x-ray
  if (!m_scheduler)
  {
    ROVER_ERROR("Error - Rover::execute: Execute called before adding a dataset");
  }

  update_camera();
  update_ray_generator();
  m_scheduler->trace_rays();
}

void
Rover::about()
{
  std::cout<<"rover version: xx.xx.xx\n";
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

void Rover::to_blueprint(conduit::Node &dataset)
{
#ifdef ROVER_PARALLEL
  // TODO: Support writing in parallel
  if(m_rank != 0)
  {
    return;
  }
#endif
  m_scheduler->to_blueprint(dataset);
}

void
Rover::save_png(const std::string &filename)
{
#ifdef ROVER_PARALLEL
  // TODO: Support writing in parallel
  if(m_rank != 0)
  {
    return;
  }
#endif
  m_scheduler->save_png(filename);
}

void
Rover::save_bov(const std::string &filename)
{
#ifdef ROVER_PARALLEL
  // TODO: Support writing in parallel
  if(m_rank != 0)
  {
    return;
  }
#endif
  m_scheduler->save_bov(filename);
}

}; //namespace rover
