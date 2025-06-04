//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_h
#define rover_h

// std includes
#include "ray_generators/camera_generator.hpp"
#include <memory>

// tpl includes
#include <conduit.hpp>
#include <vtkm_typedefs.hpp>
#include <vtkh/DataSet.hpp>

// rover includes
#include <rover_exports.h>
#include <rover_config.h>
#include <image.hpp>
#include <ray_generators/ray_generator.hpp>

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

  // TODO: Investigate if these can be guarded with #ifdef ROVER_PARALLEL
  void set_mpi_comm_handle(int mpi_comm_id);
  int  get_mpi_comm_handle();

  void add_data_set(vtkh::DataSet &);
  void update_camera();
  void update_ray_generator();
  void update_precision();

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
  void get_result(Image<vtkm::Float32> &image);
  void get_result(Image<vtkm::Float64> &image);
private:
  Node m_settings;
  vtkmCamera m_camera;
  CameraGenerator m_ray_generator;
  
  class InternalsType;
  std::shared_ptr<InternalsType> m_internals;
};

}; // namespace rover

#endif
