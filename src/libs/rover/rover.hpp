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
#include <rover_types.hpp>
#include <image.hpp>
#include <ray_generators/ray_generator.hpp>

namespace rover
{

class ROVER_API Rover
{
public:
  vtkmCamera camera;
  CameraGenerator camera_generator;
  
  Rover();
  ~Rover();

  void set_mpi_comm_handle(int mpi_comm_id);
  int  get_mpi_comm_handle();

  void add_data_set(vtkh::DataSet &);
  void set_render_settings(const RenderSettings render_settings);
  void set_ray_generator(RayGenerator *);
  void set_image_dims(int width, int height);
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
  void initialize_camera_generator();
private:
  void initialize_camera();
  class InternalsType;
  std::shared_ptr<InternalsType> m_internals;

protected:
  // Rover settings
  RenderMode render_mode;
  ScatteringType scattering_type;
  RayScope ray_scope;
  vtkmColorTable color_table;
  std::string primary_field;
  std::string secondary_field;

  // Energy settings
  bool divide_abs_by_emission;
  float unit_scalar;

  // Volume settings
  int num_samples; // approximate number of samples per ray
  vtkmRange scalar_range;

  // Camera settings
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
