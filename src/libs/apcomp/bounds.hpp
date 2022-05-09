//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_BOUNDS_HPP
#define APCOMP_BOUNDS_HPP

#include <apcomp/apcomp_exports.h>

namespace apcomp
{

struct APCOMP_API Bounds
{
  int m_min_x = 0;
  int m_max_x = 0;
  int m_min_y = 0;
  int m_max_y = 0;
};

} //namespace  apcomp
#endif
