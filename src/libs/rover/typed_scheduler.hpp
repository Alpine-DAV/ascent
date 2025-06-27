//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ROVER_TYPED_SCHEDULER_H
#define ROVER_TYPED_SCHEDULER_H

// tpl includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <vtkh/compositing/PartialCompositor.hpp>
#include <vtkh/rendering/PartialComposite.hpp>

// mpi include
#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif

// rover includes
#include <domain.hpp>
#include <image.hpp>
#include <png_utils/ascent_png_encoder.hpp>
#include <ray_generators/camera_generator.hpp>
#include <rover_exceptions.hpp>
#include <scheduler.hpp>

using namespace conduit;

namespace rover
{

template<typename FloatType>
class TypedScheduler : public Scheduler
{
public:
  TypedScheduler();

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
  void set_global_range_and_bounds();
  void add_partial(vtkhRayTracing::PartialComposite<FloatType> &partial);
  void composite();
};

}; // namespace rover
#endif
