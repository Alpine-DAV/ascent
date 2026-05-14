//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_volume_engine_h
#define rover_volume_engine_h
#include <rover_config.h>
#include <engine.hpp>
#include <viskores/rendering/ConnectivityProxy.h>

namespace rover
{

#if 0 // removing volume renderer
class VolumeEngine : public Engine
{
protected:
  viskores::rendering::ConnectivityProxy *m_tracer;
  int m_num_samples;
public:
  VolumeEngine();
  ~VolumeEngine();

  viskoresColorMap correct_opacity();
  void set_data_set(viskores::cont::DataSet &) override;
  PartialVector32 partial_trace(Ray32 &rays) override;
  PartialVector64 partial_trace(Ray64 &rays) override;
  void init_rays(Ray32 &rays) override;
  void init_rays(Ray64 &rays) override;
  void set_primary_range(const viskoresRange &range) override;
  void set_primary_field(const std::string &primary_field) override;
  void set_composite_background(bool on) override;
  void set_samples(const viskores::Bounds &global_bounds, const int &samples) override;
  viskoresRange get_primary_range() override;
  int get_num_channels() override;
};
#endif

}; // namespace rover
#endif
