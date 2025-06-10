//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_scheduler_base_h
#define rover_scheduler_base_h

#include "ray_generators/ray_generator.hpp"
#include <rover_config.h>
#include <image.hpp>
#include <vtkm_typedefs.hpp>
#include <conduit.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

using namespace conduit;

namespace rover
{

// Exists for type erasure purposes
class Scheduler
{
public:
  virtual ~Scheduler() = default;
  virtual void add_data_set(vtkmDataSet &data_set) = 0;
  virtual void trace_rays() = 0;
  virtual void set_ray_generator(RayGenerator *ray_generator) = 0;
  virtual void save_png(std::string file_name) = 0;
  virtual void save_bov(std::string file_name) = 0;
  virtual void to_blueprint(Node &dataset) = 0;
  
#ifdef ROVER_PARALLEL
  virtual void set_comm_handle(MPI_Comm comm_handle) = 0;
#endif
};

}; // namespace rover
#endif
