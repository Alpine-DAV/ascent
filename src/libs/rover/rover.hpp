//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_h
#define rover_h

// std includes
#include <memory>

// tpl includes
#include <conduit.hpp>
#include <vtkm_typedefs.hpp>

// rover includes
#include <rover_exports.h>
#include <rover_config.h>
#include <rover_types.hpp>
#include <image.hpp>
#include <ray_generators/ray_generator.hpp>

namespace rover
{

class ROVER_API Rover
{
public:
  vtkmCamera camera;
  
  Rover();
  ~Rover();

  void set_mpi_comm_handle(int mpi_comm_id);
  int  get_mpi_comm_handle();

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
  vtkmCamera& get_camera();
  
private:
  void initialize_camera();
  class InternalsType;
  std::shared_ptr<InternalsType> m_internals;

protected:
  // Default camera params
  double normal[3];
  double focus[3];
  double viewUp[3];
  double viewAngle;
  double parallelScale;     // view height
  double viewWidthOverride; // view width
  bool nonSquarePixels;
  double nearPlane;
  double farPlane;
  double imagePan[2];
  double imageZoom;
  bool perspective;
  int imageSize[2];
};

}; // namespace rover

#endif
