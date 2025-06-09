//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_engine_h
#define rover_engine_h

#include <vtkm/rendering/ConnectivityProxy.h>
#include <vtkm/cont/ColorTable.h>

#include <rover_config.h>
#include <vtkm_typedefs.hpp>
#include <settings.hpp>

namespace rover
{

struct ArraySizeFunctor
{
  vtkm::Id  *m_size;
  ArraySizeFunctor(vtkm::Id *size)
    : m_size(size)
  {}
  
  template<typename T, typename Storage>
  void operator()(const vtkm::cont::ArrayHandle<T, Storage> &array) const
  {
    *m_size = array.GetNumberOfValues();
  } //operator
};

class Engine
{
public:
  Engine();
  ~Engine();

  void validate_tracer();
  void set_data_set(vtkmDataSet &);
  void init();
  void set_primary_field();
  void set_secondary_field();
  void init_rays(Ray32 &rays);
  void init_rays(Ray64 &rays);
  PartialVector32 partial_trace(Ray32 &rays);
  PartialVector64 partial_trace(Ray64 &rays);
  int  get_num_channels();
  vtkmRange get_primary_range();
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);
  void set_color_map(const vtkmColorTable &color_map, int samples = 1024);

protected:
  vtkmDataSet m_data_set;
  vtkm::rendering::ConnectivityProxy *m_tracer;
  vtkmColorMap m_color_map;

  template<typename Precision>
  void init_emission(vtkm::rendering::raytracing::Ray<Precision> &rays,
                     const int num_bins);
};

}; // namespace rover
#endif
