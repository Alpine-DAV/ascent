//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <apcomp/apcomp.hpp>

#include <iostream>

using namespace std;


//-----------------------------------------------------------------------------
TEST(apcomp_smoke, apcomp_about)
{
  std::cout<<apcomp::about();
}

