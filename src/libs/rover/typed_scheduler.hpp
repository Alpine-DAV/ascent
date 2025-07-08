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
#include <ray_generators/vtkm_ray_generator.hpp>
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

  void add_dataset(vtkh::DataSet &dataset) override;
  void set_ray_generator(RayGenerator *ray_generator) override;
  void trace_rays() override;
  void save_png(std::string file_name) override;
  void save_bov(std::string file_name) override;
  void to_blueprint(Node &dataset) override;

  void write_blueprint_imaging_plane(Node &data_out,
                                     const std::string plane_name,
                                     const double plane_width,
                                     const double plane_height,
                                     const vtkmVec3f &center,
                                     const vtkmVec3f &left,
                                     const vtkmVec3f &up,
                                     vtkmVec3f &llc,
                                     vtkmVec3f &lrc,
                                     vtkmVec3f &ulc,
                                     vtkmVec3f &urc);

  void write_blueprint_ray_corners_mesh(Node &data_out,
                                        const vtkmVec3f &llc_near,
                                        const vtkmVec3f &llc_far,
                                        const vtkmVec3f &lrc_near,
                                        const vtkmVec3f &lrc_far,
                                        const vtkmVec3f &urc_near,
                                        const vtkmVec3f &urc_far,
                                        const vtkmVec3f &ulc_near,
                                        const vtkmVec3f &ulc_far);

  void write_blueprint_rays_mesh(Node &data_out,
                                 const int64 image_width,
                                 const int64 image_height,
                                 const double detector_width,
                                 const double detector_height,
                                 const vtkmVec3f &lrc_near,
                                 const double far_detector_width,
                                 const double far_detector_height,
                                 const vtkmVec3f &lrc_far,
                                 const vtkmVec3f &left,
                                 const vtkmVec3f &up);

protected:
  std::vector<Domain>                       m_domains;
  RayGenerator                             *m_ray_generator;
  std::vector<vtkm::Float64>                m_background;
  Image<FloatType>                          m_result;
  std::vector<PartialImage<FloatType>>      m_partial_images;

#ifdef ROVER_PARALLEL
  MPI_Comm                                  m_comm_handle;
#endif

  void create_background(const int num_channels);
  int  get_global_channels();
  void set_global_range_and_bounds();
  void add_partial(const vtkhRayTracing::PartialComposite<FloatType> &partial);
  void composite();
  template<typename PartialType>
  void typed_composite();
};

}; // namespace rover
#endif
