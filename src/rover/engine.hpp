//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_engine_h
#define rover_engine_h
#include <rover_types.hpp>
#include <vtkm_typedefs.hpp>

#include <vtkm/cont/ColorTable.h>
namespace rover {

class Engine
{
public:
  Engine(){};
  virtual ~Engine(){};

  virtual void set_data_set(vtkmDataSet &) = 0;
  virtual PartialVector32 partial_trace(Ray32 &rays) = 0;
  virtual PartialVector64 partial_trace(Ray64 &rays) = 0;
  virtual void init_rays(Ray32 &rays) = 0;
  virtual void init_rays(Ray64 &rays) = 0;
  virtual void set_primary_range(const vtkmRange &range) = 0;
  virtual void set_composite_background(bool on) = 0;
  virtual vtkmRange get_primary_range() = 0;
  virtual int get_num_channels() = 0;

  virtual void set_primary_field(const std::string &primary_field) = 0;

  virtual void set_samples(const vtkm::Bounds &global_bounds, const int &samples)
  {
    (void)samples;
    (void)global_bounds;
  }

  virtual void set_secondary_field(const std::string &secondary_field)
  {
    m_secondary_field = secondary_field;
  }

  virtual void set_color_table(const vtkmColorTable &color_map, int samples = 1024)
  {
    constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::UInt8, 4>> temp;

    color_map.Sample(samples, temp);
    m_color_map.Allocate(samples);
    auto portal = m_color_map.WritePortal();
    auto colorPortal = temp.ReadPortal();

    for (vtkm::Id i = 0; i < samples; ++i)
    {
      auto color = colorPortal.Get(i);
      vtkm::Vec<vtkm::Float32, 4> t(color[0] * conversionToFloatSpace,
                                    color[1] * conversionToFloatSpace,
                                    color[2] * conversionToFloatSpace,
                                    color[3] * conversionToFloatSpace);
      portal.Set(i, t);
    }

  }

  void set_color_map(const vtkmColorMap &color_map)
  {
    m_color_map = color_map;
  }

  vtkmColorMap get_color_map() const
  {
    return m_color_map;
  }
protected:
  vtkmColorMap m_color_map;
  std::string m_primary_field;
  std::string m_secondary_field;
};

}; // namespace rover
#endif
