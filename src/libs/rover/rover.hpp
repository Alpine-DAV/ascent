//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_h
#define rover_h

#include <rover_config.h>

#include <image.hpp>
#include <rover_exports.h>
#include <rover_types.hpp>
#include <ray_generators/ray_generator.hpp>
// vtk-m includes
#include <vtkm_typedefs.hpp>

// std includes
#include <memory>
#include <conduit.hpp>

namespace rover {

class ROVER_API Rover
{
public:
  Rover();
  ~Rover();

  void set_mpi_comm_handle(int mpi_comm_id);
  int  get_mpi_comm_handle();

  void finalize();

  void add_data_set(vtkmDataSet &);
  void set_render_settings(const RenderSettings render_settings);
  void set_ray_generator(RayGenerator *);
  void clear_data_sets();
  void set_background(const std::vector<vtkm::Float32> &background);
  void set_background(const std::vector<vtkm::Float64> &background);
  void execute();
  void about();
  void save_png(const std::string &file_name);
  void to_blueprint(conduit::Node &dataset);
  void save_png(const std::string &file_name,
                const float min_val,
                const float max_val,
                const bool log_scale);
  void save_bov(const std::string &file_name);
  void set_tracer_precision32();
  void set_tracer_precision64();
  void get_result(Image<vtkm::Float32> &image);
  void get_result(Image<vtkm::Float64> &image);
private:
  class InternalsType;
  std::shared_ptr<InternalsType> m_internals;
};

}; // namespace rover

#endif
