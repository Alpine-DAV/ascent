//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_empty_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
TEST(ascent_empty_runtime, test_empty_runtime)
{
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    Node &hello = actions.append();
    hello["action"]   = "hello!";
    actions.print();

    // we want the "empty" example pipeline
    Node open_opts;
    open_opts["runtime/type"] = "empty";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}

