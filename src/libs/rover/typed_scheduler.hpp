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
#include <ray_generators/ray_generator.hpp>
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
                                     const viskoresVec3f &center,
                                     const viskoresVec3f &left,
                                     const viskoresVec3f &up,
                                     viskoresVec3f &llc,
                                     viskoresVec3f &lrc,
                                     viskoresVec3f &ulc,
                                     viskoresVec3f &urc);

  void write_blueprint_ray_corners_mesh(Node &data_out,
                                        const viskoresVec3f &llc_near,
                                        const viskoresVec3f &llc_far,
                                        const viskoresVec3f &lrc_near,
                                        const viskoresVec3f &lrc_far,
                                        const viskoresVec3f &urc_near,
                                        const viskoresVec3f &urc_far,
                                        const viskoresVec3f &ulc_near,
                                        const viskoresVec3f &ulc_far);

  void write_blueprint_rays_mesh(Node &data_out,
                                 const int64 image_width,
                                 const int64 image_height,
                                 const double detector_width,
                                 const double detector_height,
                                 const viskoresVec3f &lrc_near,
                                 const double far_detector_width,
                                 const double far_detector_height,
                                 const viskoresVec3f &lrc_far,
                                 const viskoresVec3f &left,
                                 const viskoresVec3f &up);

protected:
  int                                       m_num_local_domains;
  bool                                      m_has_emission;
  std::vector<Domain>                       m_domains;
  RayGenerator                             *m_ray_generator;
  std::vector<viskores::Float64>                m_background;
  Image<FloatType>                          m_result;
  std::vector<PartialImage<FloatType>>      m_partial_images;

#ifdef ROVER_PARALLEL
  MPI_Comm                                  m_comm_handle;
#endif

  void create_background(const int num_energy_groups);
  int  get_global_num_energy_groups();
  void set_global_range_and_bounds();
  void add_partial(const vtkhRayTracing::PartialComposite<FloatType> &partial);
  void composite();
  template<typename PartialType>
  void typed_composite();
};

}; // namespace rover
#endif
