//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ROVER_H
#define ROVER_H

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
#include <scheduler.hpp>

using namespace conduit;

namespace rover
{

class ROVER_API Rover
{
public:
  Rover();
  ~Rover();

  void update_time_and_cycle(const Node &metadata);
  void update_settings(const Node &params);
  void print_settings();

  #ifdef ROVER_PARALLEL
  void set_mpi_comm_handle(int comm_handle);
  int  get_mpi_comm_handle();
  #endif

  void create_scheduler();
  void add_dataset(vtkh::DataSet &dataset);
  void update_camera();
  void update_ray_generator();
  void execute();

  void about();
  void save_png(const std::string &filename);
  void to_blueprint(conduit::Node &dataset);
  void save_bov(const std::string &filename);
private:
  vtkmCamera m_camera;
  CameraGenerator m_ray_generator;
  Scheduler *m_scheduler;

#ifdef ROVER_PARALLEL
  MPI_Comm m_comm_handle;
  int m_rank;
  int m_num_ranks;
#endif
};

}; // namespace rover

#endif
