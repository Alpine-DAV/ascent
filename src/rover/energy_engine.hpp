//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_energy_engine_h
#define rover_energy_engine_h

#include <engine.hpp>
#include <vtkm/rendering/ConnectivityProxy.h>
namespace rover {

class EnergyEngine : public Engine
{
protected:
  vtkmDataSet m_data_set;
  vtkm::rendering::ConnectivityProxy *m_tracer;
  vtkm::Float32 m_unit_scalar;

  int detect_num_bins();
  template<typename Precision>
  void init_emission(vtkm::rendering::raytracing::Ray<Precision> &rays,
                     const int num_bins);
public:
  EnergyEngine();
  ~EnergyEngine();

  void set_data_set(vtkm::cont::DataSet &) override;
  PartialVector32 partial_trace(Ray32 &rays) override;
  PartialVector64 partial_trace(Ray64 &rays) override;
  void init_rays(Ray32 &rays) override;
  void init_rays(Ray64 &rays) override;
  void set_primary_range(const vtkmRange &range) override;
  void set_primary_field(const std::string &primary_field) override;
  void set_secondary_field(const std::string &field) override;
  void set_composite_background(bool on) override;
  void set_unit_scalar(vtkm::Float32 unit_scalar);
  vtkmRange get_primary_range() override;
  int get_num_channels() override;
};

}; // namespace rover
#endif
