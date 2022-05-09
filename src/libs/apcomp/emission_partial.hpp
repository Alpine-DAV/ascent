//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_EMISSION_PARTIAL_HPP
#define APCOMP_EMISSION_PARTIAL_HPP

#include <assert.h>

namespace apcomp {

template<typename FloatType>
struct EmissionPartial
{
  typedef FloatType ValueType;

  int                    m_pixel_id;
  double                 m_depth;
  std::vector<FloatType> m_bins;
  std::vector<FloatType> m_emission_bins;

  EmissionPartial()
    : m_pixel_id(0),
      m_depth(0.f)
  {

  }

  void alter_bin(int bin, FloatType value)
  {
    m_bins[bin] = value;
    m_emission_bins[bin] = value;
  }

  void print()
  {
    std::cout<<"Partial id "<<m_pixel_id<<"\n";
    std::cout<<"Absorption : ";
    for(int i = 0; i < m_bins.size(); ++i)
    {
      std::cout<<m_bins[i]<<" ";
    }
    std::cout<<"\n";
    std::cout<<"Emission: ";
    for(int i = 0; i < m_bins.size(); ++i)
    {
      std::cout<<m_emission_bins[i]<<" ";
    }
    std::cout<<"\n";
  }

  bool operator < (const EmissionPartial<FloatType> &other) const
  {
    if(m_pixel_id != other.m_pixel_id)
    {
      return m_pixel_id < other.m_pixel_id;
    }
    else
    {
      return m_depth < other.m_depth;
    }
  }

  inline void blend_absorption(const EmissionPartial<FloatType> &other)
  {
    const int num_bins = static_cast<int>(m_bins.size());
    assert(num_bins == (int)other.m_bins.size());
    for(int i = 0; i < num_bins; ++i)
    {
      m_bins[i] *= other.m_bins[i];
    }
  }

  inline void blend_emission(EmissionPartial<FloatType> &other)
  {
    const int num_bins = static_cast<int>(m_bins.size());
    assert(num_bins == (int)other.m_bins.size());
    for(int i = 0; i < num_bins; ++i)
    {
      m_emission_bins[i] *= other.m_bins[i];
    }
  }

  inline void add_emission(EmissionPartial<FloatType> &other)
  {
    const int num_bins = static_cast<int>(m_bins.size());
    assert(num_bins == (int)other.m_bins.size());
    for(int i = 0; i < num_bins; ++i)
    {
      m_emission_bins[i] += other.m_emission_bins[i];
    }
  }

  static void composite_background(std::vector<EmissionPartial> &partials,
                                   const std::vector<FloatType> &background)
  {
    //for(
  }

};

} // namespace rover


#endif
