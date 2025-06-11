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
#include <settings.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

namespace rover
{

// Namespaced settings instance
Node settings;

Rover::Rover()
{
  // Ensure that we always start from a default state
  rover::settings.reset();
  // TODO: Figure out if color_table needs to be set by us here
  rover::settings["rover/color_table"] = "Cool to Warm";
  rover::settings["rover/divide_emission_by_abs"] = "false";
  rover::settings["rover/height"] = 200;
  rover::settings["rover/num_samples"] = 400;
  rover::settings["rover/precision"] = "single";
  rover::settings["rover/ray_scope"] = "global_rays";
  rover::settings["rover/scattering_type"] = "non_scattering";
  rover::settings["rover/width"] = 200;
  rover::settings["rover/unit_scalar"] = 1.0f;

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
Rover::update_settings(Node &params)
{
  ROVER_INFO("Executing Rover::update_settings");

  if (params.has_child("rover"))
  {
    for (const auto &param_name : params["rover"].child_names())
    {
      rover::settings["rover"][param_name].set(params["rover"][param_name]);
    }
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
  const std::string precision = rover::settings["rover/precision"].as_string();
  if ("double" == precision)
  {
    m_scheduler = new Scheduler<vtkm::Float64>();
  }
  else // ("single" == precision)
  {
    m_scheduler = new Scheduler<vtkm::Float32>();
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
Rover::add_data_set(vtkh::DataSet &dataset)
{
  ROVER_INFO("Executing Rover::add_data_set");
  // The scheduler needs to be created before data can be added
  // to it, else we segfault
  if (!m_scheduler)
  {
    create_scheduler();
  }

  for (int i = 0; i < dataset.GetNumberOfDomains(); i++)
  {
    m_scheduler->add_data_set(dataset.GetDomain(i));
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

  // TODO: Change each instance of has_path to has_child
  if (rover::settings.has_path("camera/zoom"))
  {
    float64 image_zoom = rover::settings["camera/zoom"].value();
    m_camera.Zoom(log(image_zoom) / log(4.0));
  }

  if (rover::settings.has_path("camera/look_at"))
  {
    float64_accessor vec3 = rover::settings["camera/look_at"].value();
    vtkmVec3f look_at(vec3[0], vec3[1], vec3[2]);
    m_camera.SetLookAt(look_at);
  }
  
  if (rover::settings.has_path("camera/up"))
  {
    float64_accessor vec3 = rover::settings["camera/up"].value();
    vtkmVec3f up(vec3[0], vec3[1], vec3[2]);
    m_camera.SetViewUp(up);
  }
  
  if (rover::settings.has_path("camera/fov"))
  {
    float64 fov = rover::settings["camera/fov"].value();
    m_camera.SetFieldOfView(fov);
  }
  
  if (rover::settings.has_path("camera/xpan") || rover::settings.has_path("camera/ypan"))
  {
    float64 xpan = 0.0;
    float64 ypan = 0.0;

    if (rover::settings.has_path("camera/xpan"))
    {
      xpan = rover::settings["camera/xpan"].value();
    }

    if (rover::settings.has_path("camera/ypan"))
    {
      ypan = rover::settings["camera/ypan"].value();
    }
    
    m_camera.Pan(xpan, ypan);
  }
  
  if (rover::settings.has_path("camera/near_plane") || rover::settings.has_path("camera/far_plane"))
  {
    vtkm::Range clipping_range;

    if (rover::settings.has_path("camera/near_plane"))
    {
      clipping_range.Min = rover::settings["camera/near_plane"].value();
    }
    
    if (rover::settings.has_path("camera/far_plane"))
    {
      clipping_range.Max = rover::settings["camera/far_plane"].value();
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
Rover::set_background(const std::vector<vtkm::Float32> &background)
{
  m_scheduler->set_background(background);
}

void
Rover::set_background(const std::vector<vtkm::Float64> &background)
{
  m_scheduler->set_background(background);
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
Rover::save_png(const std::string &file_name)
{
#ifdef ROVER_PARALLEL
  // TODO: Support writing in parallel
  if(m_rank != 0)
  {
    return;
  }
#endif
  m_scheduler->save_result(file_name);
}

void
Rover::save_png(const std::string &file_name,
                const float min_val,
                const float max_val,
                const bool log_scale)
{
#ifdef ROVER_PARALLEL
  // TODO: Support writing in parallel
  if(m_rank != 0)
  {
    return;
  }
#endif
  m_scheduler->save_result(file_name, min_val, max_val, log_scale);
}

void
Rover::save_bov(const std::string &file_name)
{
  m_scheduler->save_bov(file_name);
}

void
Rover::get_result(Image<vtkm::Float32> &image)
{
  m_scheduler->get_result(image);
}

void
Rover::get_result(Image<vtkm::Float64> &image)
{
  m_scheduler->get_result(image);
}

}; //namespace rover
