//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_scheduler_h
#define rover_scheduler_h

#include <rover_config.h>
#include <domain.hpp>
#include <image.hpp>
#include <engine.hpp>
#include <scheduler.hpp>
#include <ray_generators/ray_generator.hpp>
#include <vtkm_typedefs.hpp>
#include <conduit.hpp>
#include <settings.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

using namespace conduit;

namespace rover
{

template<typename FloatType>
class SchedulerImpl : public Scheduler
{
public:
  SchedulerImpl();
  ~SchedulerImpl();

#ifdef ROVER_PARALLEL
  void set_comm_handle(MPI_Comm comm_handle) override;
#endif

  void add_dataset(vtkmDataSet &dataset) override;
  void set_ray_generator(RayGenerator *ray_generator) override;
  void trace_rays() override;
  void save_png(std::string file_name) override;
  void save_bov(std::string file_name) override;
  void to_blueprint(Node &dataset) override;

protected:
  std::vector<Domain>                       m_domains;
  RayGenerator                             *m_ray_generator;
  std::vector<vtkm::Float64>                m_background;
  Image<FloatType>                          m_result;
  std::vector<PartialImage<FloatType>>      m_partial_images;

#ifdef ROVER_PARALLEL
  MPI_Comm                                  m_comm_handle;
#endif

  void create_default_background(const int num_channels);
  void set_background(const std::vector<vtkm::Float32> &background);
  void set_background(const std::vector<vtkm::Float64> &background);

  int  get_global_channels();
  void set_global_scalar_range();
  void set_global_bounds();
  void add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial);
  void composite();
};

}; // namespace rover
#endif
