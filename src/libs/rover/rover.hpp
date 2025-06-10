//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_h
#define rover_h

// std includes

// tpl includes
#include <conduit.hpp>
#include <vtkm_typedefs.hpp>
#include <vtkh/DataSet.hpp>

// mpi include
#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

// rover includes
#include <rover_exports.h>
#include <rover_config.h>
#include <settings.hpp>
#include <image.hpp>
#include <ray_generators/ray_generator.hpp>
#include <ray_generators/camera_generator.hpp>
#include <scheduler_base.hpp>

using namespace conduit;

namespace rover
{

class ROVER_API Rover
{
public:
  Rover();
  ~Rover();

  void update_settings(Node &params);
  void print_settings();

  #ifdef ROVER_PARALLEL
  void set_mpi_comm_handle(int comm_handle);
  int  get_mpi_comm_handle();
  #endif

  SchedulerBase & create_scheduler();
  void add_data_set(SchedulerBase &scheduler, vtkh::DataSet &dataset);
  void update_camera();
  void update_ray_generator(SchedulerBase &scheduler);
  void execute(SchedulerBase &scheduler);

  void about();
  void save_png(SchedulerBase &scheduler, const std::string &file_name);
  void to_blueprint(SchedulerBase &scheduler, conduit::Node &dataset);
  void save_png(SchedulerBase &scheduler, 
                const std::string &file_name,
                const float min_val,
                const float max_val,
                const bool log_scale);
  void save_bov(SchedulerBase &scheduler, const std::string &file_name);
  void get_result(SchedulerBase &scheduler, Image<vtkm::Float32> &image);
  void get_result(SchedulerBase &scheduler, Image<vtkm::Float64> &image);
private:
  vtkmCamera m_camera;
  CameraGenerator m_ray_generator;

#ifdef ROVER_PARALLEL
  MPI_Comm m_comm_handle;
  int m_rank;
  int m_num_ranks;
#endif
};

}; // namespace rover

#endif
