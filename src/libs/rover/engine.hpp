//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_engine_h
#define rover_engine_h

#include <vector>
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
  void set_dataset(const vtkmDataSet &dataset);
  void set_num_channels(const int num_channels);
  int get_num_channels();

  void partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                     std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float32>> &partials,
                     const vtkm::cont::ArrayHandle<vtkm::Float32> &background);

  void partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                     std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float64>> &partials,
                     const vtkm::cont::ArrayHandle<vtkm::Float64> &background);

  vtkmRange get_primary_range();
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);

protected:
  vtkh::rendering::ConnectivityProxy *m_tracer;
  int m_num_channels;
};

}; // namespace rover
#endif
