//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_scheduler_h
#define rover_scheduler_h

#include <domain.hpp>
#include <image.hpp>
#include <engine.hpp>
#include <scheduler_base.hpp>
#include <rover_types.hpp>
#include <ray_generators/ray_generator.hpp>
#include <vtkm_typedefs.hpp>

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

template<typename FloatType>
class Scheduler : public SchedulerBase
{
public:
  Scheduler();
  virtual ~Scheduler();
  void trace_rays() override;
  void save_result(std::string file_name) override;
  void save_result(std::string file_name,
                   float min_val,
                   float max_val,
                   bool log_scale) override;
  void save_bov(std::string file_name) override;
  virtual void to_blueprint(conduit::Node &dataset) override;

  virtual void get_result(Image<vtkm::Float32> &image) override;
  virtual void get_result(Image<vtkm::Float64> &image) override;
protected:
  void composite();
  void set_global_scalar_range();
  void set_global_bounds();
  int  get_global_channels();
  Image<FloatType>                          m_result;
  std::vector<PartialImage<FloatType>>      m_partial_images;

  void add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial, int width, int height);
private:

};

}; // namespace rover
#endif
