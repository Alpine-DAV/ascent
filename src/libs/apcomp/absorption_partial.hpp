//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_ABSORPTION_PARTIAL_HPP
#define APCOMP_ABSORPTION_PARTIAL_HPP

#include <assert.h>

namespace apcomp {

template<typename FloatType>
struct AbsorptionPartial
{
  typedef FloatType ValueType;
  int                    m_pixel_id;
  double                 m_depth;
  std::vector<FloatType> m_bins;

  AbsorptionPartial()
    : m_pixel_id(0),
      m_depth(0.f)
  {}

  void print()
  {

  }

  bool operator < (const AbsorptionPartial<FloatType> &other) const
  {
    //
    // In absorption only we can blend the same
    // pixel ids in any order
    //
    return m_pixel_id < other.m_pixel_id;
  }

  inline void blend(const AbsorptionPartial<FloatType> &other)
  {
    const int num_bins = static_cast<int>(m_bins.size());
    assert(num_bins == (int)other.m_bins.size());
    for(int i = 0; i < num_bins; ++i)
    {
      m_bins[i] *= other.m_bins[i];
    }
  }

  static void composite_background(std::vector<AbsorptionPartial> &partials,
                                   const std::vector<FloatType> &background)
  {
    const int size = static_cast<int>(partials.size());
    AbsorptionPartial<FloatType> bg;
    bg.m_bins = background;
#ifdef APCOMP_USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      partials[i].blend(bg);
    }
  }

};

} // namespace rover


#endif
