//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_engine_h
#define rover_engine_h

#include <vtkh/rendering/ConnectivityProxy.hpp>
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
  void set_dataset(vtkmDataSet &dataset);
  void init_rays(Ray32 &rays);
  void init_rays(Ray64 &rays);
  void partial_trace(Ray32 &rays, PartialVector32 &partials);
  void partial_trace(Ray64 &rays, PartialVector64 &partials);
  int  get_num_energy_groups();
  vtkmRange get_primary_range();
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);

protected:
  vtkmDataSet m_dataset;
  vtkh::rendering::ConnectivityProxy *m_tracer;

  template<typename Precision>
  void init_emission(vtkmRayTracing::Ray<Precision> &rays,
                     const int num_bins);
};

}; // namespace rover
#endif
