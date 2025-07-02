//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ROVER_SCHEDULER_H
#define ROVER_SCHEDULER_H

// tpl includes
#include <conduit.hpp>
#include <vtkh/DataSet.hpp>

// rover includes
#include "ray_generators/ray_generator.hpp"

// mpi include
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

#ifdef ROVER_PARALLEL
  virtual void set_comm_handle(MPI_Comm comm_handle) = 0;
#endif

  virtual void add_dataset(vtkh::DataSet &dataset) = 0;
  virtual void set_ray_generator(RayGenerator *ray_generator) = 0;
  virtual void trace_rays() = 0;
  virtual void save_png(std::string file_name) = 0;
  virtual void save_bov(std::string file_name) = 0;
  virtual void to_blueprint(Node &dataset) = 0;
};

}; // namespace rover
#endif
