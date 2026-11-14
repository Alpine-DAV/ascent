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

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
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

    // Exercise the exception-safe region and typed value helpers. Declare
    // values before the scope so they are present on its end snapshot.
    {
        ascent::annotations::Value input_cells("test.input.cells");
        ascent::annotations::Value output_cells("test.output.cells");
        input_cells.set(std::uint64_t(10));
        {
            ASCENT_ANNOTATE_MARK_SCOPE("test_caliper_scope");
            output_cells.set(std::uint64_t(7));
        }
    }

    // Exercise MARK_SCOPE cleanup when an exception is thrown.
    try
    {
        ASCENT_ANNOTATE_MARK_SCOPE("test_exception_scope");
        throw std::runtime_error("exercise annotation scope unwinding");
    }
    catch(const std::runtime_error &)
    {
    }

    // VTK-h can link multiple filters. Verify that scoped values
    // with the same attribute names restore the outer values 
    // when the inner values are destroyed
    {
        ascent::annotations::Value outer_name("test.nested.name");
        ascent::annotations::Value outer_count("test.nested.count");
        outer_name.set("outer");
        outer_count.set(std::uint64_t(10));
        ASCENT_ANNOTATE_MARK_SCOPE("test_nested_outer");
        {
            ascent::annotations::Value inner_name("test.nested.name");
            ascent::annotations::Value inner_count("test.nested.count");
            inner_name.set("inner");
            inner_count.set(std::uint64_t(5));
            ASCENT_ANNOTATE_MARK_SCOPE("test_nested_inner");
        }
    }
    {
        ASCENT_ANNOTATE_MARK_SCOPE("test_after_nested_values");
    }

    ascent::annotations::flush();
    ascent::annotations::finalize();
  
    if(ascent::annotations::supported())
    {
        // make sure perf output file exists
        EXPECT_TRUE(conduit::utils::is_file(tout_file));

        std::ifstream report_stream(tout_file.c_str());
        std::stringstream report;
        report << report_stream.rdbuf();
        EXPECT_NE(report.str().find("test_after_nested_values", std::string::npos));
    }
}
