//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_scheduler_base_h
#define rover_scheduler_base_h

#include <rover_config.h>

#include <domain.hpp>
#include <image.hpp>
#include <engine.hpp>
#include <rover_types.hpp>
#include <ray_generators/ray_generator.hpp>
#include <vtkm_typedefs.hpp>
#include <conduit.hpp>

#ifdef ROVER_PARALLEL
#include <mpi.h>
#endif
//
// Scheduler types:
//  static: all ranks gets all rays
//  normal compositing -
//    back to front (energy): absorbtion, absorbtion + emmission
//    front to back (volume): normal volume rendering
//  dynamic(scattering):
//    domain passing -
//      front to back: volume rendering and ray tracing
//      back to front: both energy types.
//
//
//
namespace rover {

class SchedulerBase
{
public:
  SchedulerBase();
  virtual ~SchedulerBase();
  virtual void trace_rays() = 0;
  virtual void save_result(std::string file_name) = 0;
  virtual void save_result(std::string file_name,
                           float min_val,
                           float max_val,
                           bool log_scale) = 0;
  virtual void save_bov(std::string file_name) = 0;
  virtual void to_blueprint(conduit::Node &dataset) = 0;
  void clear_data_sets();
  //
  // Setters
  //
  void set_render_settings(const RenderSettings render_settings);
  void add_data_set(vtkmDataSet &data_set);
  void set_domains(std::vector<Domain> &domains);
  void set_ray_generator(RayGenerator *ray_generator);
  void set_background(const std::vector<vtkm::Float32> &background);
  void set_background(const std::vector<vtkm::Float64> &background);
#ifdef ROVER_PARALLEL
  void set_comm_handle(MPI_Comm comm_handle);
#endif
  //
  // Getters
  //
  std::vector<Domain> get_domains();
  RenderSettings get_render_settings() const;
  vtkmDataSet    get_data_set(const int &domain);
  virtual void get_result(Image<vtkm::Float32> &image) = 0;
  virtual void get_result(Image<vtkm::Float64> &image) = 0;
protected:
  std::vector<Domain>                       m_domains;
  RenderSettings                            m_render_settings;
  RayGenerator                             *m_ray_generator;
  std::vector<vtkm::Float64>                m_background;
  void create_default_background(const int num_channels);
#ifdef ROVER_PARALLEL
  MPI_Comm                                  m_comm_handle;
#endif

};

}; // namespace rover
#endif
