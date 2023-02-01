//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_utils.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_resources.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
TEST(ascent_utils, ascent_copy_dir)
{
    string output_path = conduit::utils::join_path(prepare_output_dir(),"my_folder");

    string idx_fpath = conduit::utils::join_path(output_path,"ascent/index.html");

    // for multiple runs of this test:
    //  we don't have a util to kill the entire dir, so
    //  we simply remove a known file, and check that the copy restores it

    if(conduit::utils::is_file(idx_fpath))
    {
        conduit::utils::remove_file(idx_fpath);
    }

    // load ascent web resources from compiled in resource tree
    Node ascent_rc;
    ascent::resources::load_compiled_resource_tree("ascent_web",
                                                    ascent_rc);
    if(ascent_rc.dtype().is_empty())
    {
        ASCENT_ERROR("Failed to load compiled resources for ascent_web");
    }

    ascent::resources::expand_resource_tree_to_file_system(ascent_rc,
                                                           output_path);

    EXPECT_TRUE(conduit::utils::is_directory(conduit::utils::join_path(output_path,"resources")));
    EXPECT_TRUE(conduit::utils::is_file(idx_fpath));
}

