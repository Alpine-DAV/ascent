// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_ascent_annotations.cpp
///
//-----------------------------------------------------------------------------


#include "ascent.hpp"
#include "ascent_annotations.hpp"

#include <iostream>
#include <limits>
#include "gtest/gtest.h"

#include "t_config.hpp"

//-----------------------------------------------------------------------------
void
annotate_test_func()
{
    ASCENT_ANNOTATE_MARK_FUNCTION;
}

//-----------------------------------------------------------------------------
TEST(ascent_utils, annotations_support)
{
    conduit::Node about;
    ascent::about(about);

    if( ascent::annotations::supported() )
    {
        EXPECT_EQ(about["annotations"].as_string(),"enabled");
    }
    else
    {
        EXPECT_EQ(about["annotations"].as_string(),"disabled");
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_utils, annotations)
{
    std::string tout_file = "tout_annotations_file.txt";
    if(ascent::annotations::supported())
    {
        // clean up output file if it exists
        conduit::utils::remove_path_if_exists(tout_file);
    }
    
    conduit::Node opts;
    opts["config"] = "runtime-report";
    opts["output_file"] = tout_file;
    ascent::annotations::initialize(opts);
    ASCENT_ANNOTATE_MARK_BEGIN("test_region");
    annotate_test_func();
    {
        ASCENT_ANNOTATE_MARK_SCOPE("test_scope");
        conduit::utils::sleep(100);
    }
    ASCENT_ANNOTATE_MARK_END("test_region");

    ascent::annotations::flush();
    ascent::annotations::finalize();
  
    if(ascent::annotations::supported())
    {
        // make sure perf output file exists
        EXPECT_TRUE(conduit::utils::is_file(tout_file));
    }
}


