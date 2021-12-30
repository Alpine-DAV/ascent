//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_flow_timercpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <flow.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"



using namespace std;
using namespace conduit;
using namespace flow;


//-----------------------------------------------------------------------------
TEST(ascent_flow_timer, time_passed)
{
    flow::Timer t;
    float delta_1 = t.elapsed();
    float delta_2 = t.elapsed();
    std::cout << "d1 " << delta_1 << " d2 " << delta_2 << std::endl;

    EXPECT_GT(delta_2,delta_1);

}
