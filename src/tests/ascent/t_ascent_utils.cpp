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
#include <ascent_metadata.hpp>
#include <ascent_string_utils.hpp>

#include <iostream>
#include <math.h>
#include <conduit.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
TEST(ascent_utils, ascent_copy_dir)
{
    string output_path = conduit::utils::join_path(prepare_output_dir(),"my_folder");

    string idx_fpath = conduit::utils::join_path(output_path,"index.html");

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

    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    ascent::resources::expand_resource_tree_to_file_system(ascent_rc,
                                                           output_path);

    EXPECT_TRUE(conduit::utils::is_directory(conduit::utils::join_path(output_path,"resources")));
    EXPECT_TRUE(conduit::utils::is_file(idx_fpath));
}

//-----------------------------------------------------------------------------
TEST(ascent_utils, ascent_string_fmt_basic)
{
    // Set up the metadata variables beforehand so there are inputs for the formatter
    Metadata::n_metadata["cycle"] = 100;
    Metadata::n_metadata["time"] = 3.1415926;

    std::string expected_result = "output_path_100_00012_3.1416";
    std::string result = ascent::expand_path_special_variables("output_path_{cycle:3d}_{family:05d}_{time:0.4f}",12);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_cycle_integer_fmt)
{
    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["cycle"] = 9349;

    std::string expected_result = "output_path_cycle_int_9349_0000000000000000000000000000000000000009349_9349_9349";
    std::string result = ascent::expand_path_special_variables("output_path_cycle_int_{cycle:3d}_{cycle:00043d}_{cycle:3i}_{cycle:3u}");
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_cycle_float_fmt)
{
    // The cycle variable will typically be an integer so this test is verifying that the cycle
    // value is being converted to floating point correctly

    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["cycle"] = 2139;

    // Test floating point formats
    std::string expected_result_float = "output_path_cycle_float_2139.00_2139.00";
    std::string result_float = ascent::expand_path_special_variables("output_path_cycle_float_{cycle:02.02f}_{cycle:02.02F}");
    std::cout << result_float << std::endl;
    EXPECT_TRUE(expected_result_float == result_float);

    // Test scientific notation format
    std::string expected_result_scientific = "output_path_cycle_scientific_2.14e+03_2.14E+03";
    std::string result_scientific = ascent::expand_path_special_variables("output_path_cycle_scientific_{cycle:02.02e}_{cycle:02.02E}");
    std::cout << result_scientific << std::endl;
    EXPECT_TRUE(expected_result_scientific == result_scientific);

    // Test g format
    std::string expected_result_g = "output_path_cycle_g_2.1e+03_2.1E+03";
    std::string result_g = ascent::expand_path_special_variables("output_path_cycle_g_{cycle:02.02g}_{cycle:02.02G}");
    std::cout << result_g << std::endl;
    EXPECT_TRUE(expected_result_g == result_g);
}

TEST(ascent_utils, ascent_string_fmt_time_float_fmt)
{
    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["time"] = 103.141592;

    // Test floating point formats
    std::string expected_result_float = "output_path_time_float_103.14_103.14";
    std::string result_float = ascent::expand_path_special_variables("output_path_time_float_{time:02.02f}_{time:02.02F}");
    std::cout << result_float << std::endl;
    EXPECT_TRUE(expected_result_float == result_float);

    // Test scientific notation format
    std::string expected_result_scientific = "output_path_time_scientific_1.03e+02_1.03E+02";
    std::string result_scientific = ascent::expand_path_special_variables("output_path_time_scientific_{time:02.02e}_{time:02.02E}");
    std::cout << result_scientific << std::endl;
    EXPECT_TRUE(expected_result_scientific == result_scientific);

    // Test g format
    std::string expected_result_g = "output_path_time_g_1e+02_1E+02";
    std::string result_g = ascent::expand_path_special_variables("output_path_time_g_{time:05.02g}_{time:02.02G}");
    std::cout << result_g << std::endl;
    EXPECT_TRUE(expected_result_g == result_g);
}

TEST(ascent_utils, ascent_string_fmt_time_integer_fmt)
{
    // The cycle variable will typically be a float so this test is verifying that the time
    // value is being converted to an integer correctly

    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["time"] = 329.324134;

    std::string expected_result = "output_path_time_int_329_0000000000000000000000000000000000000000329_329_329";
    std::string result = ascent::expand_path_special_variables("output_path_time_int_{time:3d}_{time:00043d}_{time:3i}_{time:3u}");
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_family)
{
    for (int i = 0; i < 4; i++){
        char expected_result[50];
        std::string expected_pattern = "output_path_family_%03d_%05.2f_%05.03g";
        float current_family = i+5;
        snprintf(expected_result, sizeof(expected_result), expected_pattern.c_str(), static_cast<int>(current_family), current_family, current_family);

        std::string result = ascent::expand_path_special_variables("output_path_family_{family:03d}_{family:05.2f}_{family:05.03g}", 5);
        std::cout << result << std::endl;


        EXPECT_TRUE(expected_result == result);
    }
}

TEST(ascent_utils, ascent_string_fmt_none)
{
    std::string expected_result = "output_path_none_12";
    std::string result = ascent::expand_path_special_variables("output_path_none_", 12);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_invalid_int_format)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("output_path_none_{family:12.3d}");
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn) {
        if (warn.message().find("Invalid format specifier: '12.3d'.") != std::string::npos) {
            error_occured = true;
        }
        else
        {
            std::cout << "The error that was thrown did not match the expected 'Invalid format specifier' error" << endl;
            std::cout << warn.message() << std::endl;
        }
    }

    EXPECT_TRUE(error_occured);
}

TEST(ascent_utils, ascent_string_fmt_invalid_float_format)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("output_path_none_{family:2.2.3f}");
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn) {
        if (warn.message().find("Invalid format specifier: '2.2.3f'.") != std::string::npos) {
            error_occured = true;
        }
        else
        {
            std::cout << "The error that was thrown did not match the expected 'Invalid format specifier' error" << endl;
            std::cout << warn.message() << std::endl;
        }
    }
    
    EXPECT_TRUE(error_occured);
}

TEST(ascent_utils, ascent_string_fmt_invalid_no_format)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("output_path_none_{family:}");
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn) {
        if (warn.message().find("No format specifications given.") != std::string::npos) {
            error_occured = true;
        }
        else
        {
            std::cout << "The error that was thrown did not match the expected 'No format specifications given.' error" << endl;
            std::cout << warn.message() << std::endl;
        }
    }

    EXPECT_TRUE(error_occured);
}

TEST(ascent_utils, ascent_string_fmt_invalid_keyword)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("output_path_none_{invalid:128f}");
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn) {
        if (warn.message().find("Invalid format keyword 'invalid'.") != std::string::npos) {
            error_occured = true;
        }
        else
        {
            std::cout << "The error that was thrown did not match the expected 'Invalid format keyword' error" << endl;
            std::cout << warn.message() << std::endl;
        }
    }

    EXPECT_TRUE(error_occured);
}