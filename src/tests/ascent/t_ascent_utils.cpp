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
    Metadata::n_metadata["time"] = 3.141592;
    Metadata::n_metadata["family_value_seed"] = 12;

    std::string expected_result = "t_output_path_100_00012_3.1416";
    std::string result = ascent::expand_path_special_variables("t_output_path_{cycle:3d}_{family:05d}_{time:0.4f}", "", -1);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_cycle_integer_fmt)
{
    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["cycle"] = 100;

    std::string expected_result = "t_output_path_cycle_int_100_0000000000000000000000000000000000000000100_100_100";
    std::string result = ascent::expand_path_special_variables("t_output_path_cycle_int_{cycle:3d}_{cycle:00043d}_{cycle:3i}_{cycle:3u}", "", -1);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_cycle_float_fmt)
{
    // The cycle variable will typically be an integer so this test is verifying that the cycle
    // value is being converted to floating point correctly

    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["cycle"] = 100;

    // Test floating point formats
    std::string expected_result_float = "t_output_path_cycle_float_100.00_100.00";
    std::string result_float = ascent::expand_path_special_variables("t_output_path_cycle_float_{cycle:02.02f}_{cycle:02.02F}", "", -1);
    std::cout << result_float << std::endl;
    EXPECT_TRUE(expected_result_float == result_float);

    // Test scientific notation format
    std::string expected_result_scientific = "t_output_path_cycle_scientific_1.00e+02_1.00E+02";
    std::string result_scientific = ascent::expand_path_special_variables("t_output_path_cycle_scientific_{cycle:02.02e}_{cycle:02.02E}", "", -1);
    std::cout << result_scientific << std::endl;
    EXPECT_TRUE(expected_result_scientific == result_scientific);

    // Test g format
    std::string expected_result_g = "t_output_path_cycle_g_1e+02_1E+02";
    std::string result_g = ascent::expand_path_special_variables("t_output_path_cycle_g_{cycle:02.02g}_{cycle:02.02G}", "", -1);
    std::cout << result_g << std::endl;
    EXPECT_TRUE(expected_result_g == result_g);
}

TEST(ascent_utils, ascent_string_fmt_time_float_fmt)
{
    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["time"] = 3.141592;

    // Test floating point formats
    std::string expected_result_float = "t_output_path_time_float_3.14_3.14";
    std::string result_float = ascent::expand_path_special_variables("t_output_path_time_float_{time:02.02f}_{time:02.02F}", "", -1);
    std::cout << result_float << std::endl;
    EXPECT_TRUE(expected_result_float == result_float);

    // Test scientific notation format
    std::string expected_result_scientific = "t_output_path_time_scientific_3.14e+00_3.14E+00";
    std::string result_scientific = ascent::expand_path_special_variables("t_output_path_time_scientific_{time:02.02e}_{time:02.02E}", "", -1);
    std::cout << result_scientific << std::endl;
    EXPECT_TRUE(expected_result_scientific == result_scientific);

    // Test g format
    std::string expected_result_g = "t_output_path_time_g_003.1_3.1";
    std::string result_g = ascent::expand_path_special_variables("t_output_path_time_g_{time:05.02g}_{time:02.02G}", "", -1);
    std::cout << result_g << std::endl;
    EXPECT_TRUE(expected_result_g == result_g);
}

TEST(ascent_utils, ascent_string_fmt_time_integer_fmt)
{
    // The cycle variable will typically be a float so this test is verifying that the time
    // value is being converted to an integer correctly

    // Set up the metadata variable beforehand so there is an input for the formatter
    Metadata::n_metadata["time"] = 3.141592;

    std::string expected_result = "t_output_path_time_int_003_0000000000000000000000000000000000000000003_003_003";
    std::string result = ascent::expand_path_special_variables("t_output_path_time_int_{time:03d}_{time:00043d}_{time:03i}_{time:03u}", "", -1);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_family)
{
    Metadata::n_metadata["family_value_seed"] = 0;

    for (int i = 0; i < 4; i++)
    {
        char expected_result[50];
        std::string expected_pattern = "t_output_path_family_%03d_%05.2f_%05.03g";
        snprintf(expected_result, sizeof(expected_result), expected_pattern.c_str(), i, static_cast<float>(i), static_cast<float>(i));

        std::string result = ascent::expand_path_special_variables("t_output_path_family_{family:03d}_{family:05.2f}_{family:05.03g}", "", -1);
        std::cout << expected_result << std::endl;
        std::cout << result << std::endl;

        EXPECT_TRUE(expected_result == result);
    }
}

TEST(ascent_utils, ascent_string_fmt_none)
{
    std::string expected_result = "t_output_path_none_100";
    std::string result = ascent::expand_path_special_variables("t_output_path_none_", "", -1);
    std::cout << result << std::endl;
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_invalid_int_format)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("t_output_path_none_{family:12.3d}", "", -1);
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn)
    {
        if (warn.message().find("Invalid format specifier: '12.3d'.") != std::string::npos)
        {
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
        std::string result = ascent::expand_path_special_variables("t_output_path_none_{family:2.2.3f}", "", -1);
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn)
    {
        if (warn.message().find("Invalid format specifier: '2.2.3f'.") != std::string::npos)
        {
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

TEST(ascent_utils, ascent_string_fmt_just_type)
{
    Metadata::n_metadata["cycle"] = 100;
    Metadata::n_metadata["time"] = 3.141592;
    Metadata::n_metadata["family_value_seed"] = 6;

    std::string result = ascent::expand_path_special_variables("t_output_path_none_{cycle:f}_{time:d}_{family:f}", "", -1);
    std::string expected_result = "t_output_path_none_100.000000_3_6.000000";
    std::cout << result << std::endl;
   
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_no_format)
{
    Metadata::n_metadata["cycle"] = 100;
    Metadata::n_metadata["time"] = 3.141592;
    Metadata::n_metadata["family_value_seed"] = 8;

    std::string result = ascent::expand_path_special_variables("t_output_path_none_{cycle}_{time}_{family}", "", -1);
    std::string expected_result = "t_output_path_none_100_3.141592_8";
    std::cout << result << std::endl;
   
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_no_format_colon)
{
    Metadata::n_metadata["cycle"] = 100;
    Metadata::n_metadata["time"] = 3.141592;
    Metadata::n_metadata["family_value_seed"] = 10;

    std::string result = ascent::expand_path_special_variables("t_output_path_none_{cycle:}_{time:}_{family:}", "", -1);
    std::string expected_result = "t_output_path_none_100_3.141592_10";
    std::cout << result << std::endl;
   
    EXPECT_TRUE(expected_result == result);
}

TEST(ascent_utils, ascent_string_fmt_invalid_keyword)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("t_output_path_none_{invalid:128f}", "", -1);
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn)
    {
        if (warn.message().find("Invalid format keyword 'invalid'.") != std::string::npos)
        {
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

TEST(ascent_utils, ascent_string_fmt_no_keyword)
{
    bool error_occured = false;

    try
    {
        std::string result = ascent::expand_path_special_variables("t_output_path_none_{:128f}", "", -1);
        std::cout << result << std::endl;
    }
    catch (conduit::Error &warn)
    {
        if (warn.message().find("Invalid format keyword ''.") != std::string::npos)
        {
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

TEST(ascent_utils, ascent_string_fmt_family_check_dir) {
    string output_path = prepare_output_dir();
    Metadata::n_metadata["family_value_seed"] = 0;

    string pre_existing_file_name_1 = conduit::utils::join_file_path(output_path, "t_output_path_family_check_dir_000_0012_3.14.root");
    string pre_existing_file_name_2 = conduit::utils::join_file_path(output_path, "t_output_path_family_check_dir_000_1.400000e+01_3.14.root");
    std::ofstream file_1(pre_existing_file_name_1);
    if (file_1.is_open()) {
        file_1 << "This is a fake file for testing.\n";
        file_1.close();
    }

    std::ofstream file_2(pre_existing_file_name_2);
    if (file_2.is_open()) {
        file_2 << "This is a fake file for testing.\n";
        file_2.close();
    }

    string output_file = conduit::utils::join_file_path(output_path, "t_output_path_family_check_dir_{cycle:3d}_{family:05d}_{time:0.4f}");
    
    int result = ascent::get_family_value(output_file, ".root", -1, 0);
    EXPECT_TRUE(result == 15);
    
    remove_test_file(pre_existing_file_name_1);
    remove_test_file(pre_existing_file_name_2);
}

TEST(ascent_utils, ascent_string_fmt_family_check_diff_ext) {
    string output_path = prepare_output_dir();
    Metadata::n_metadata["family_value_seed"] = 0;

    string pre_existing_file_name_1 = conduit::utils::join_file_path(output_path, "t_output_path_family_check_01000_diff_ext.root");
    string pre_existing_file_name_2 = conduit::utils::join_file_path(output_path, "t_output_path_family_check_01001_diff_ext.root");
    std::ofstream file_1(pre_existing_file_name_1);
    if (file_1.is_open()) {
        file_1 << "This is a fake file for testing.\n";
        file_1.close();
    }

    std::ofstream file_2(pre_existing_file_name_2);
    if (file_2.is_open()) {
        file_2 << "This is a fake file for testing.\n";
        file_2.close();
    }

    string output_file = conduit::utils::join_file_path(output_path, "t_output_path_family_check_{family:05d}_diff_ext");
    std::string expected_result = conduit::utils::join_file_path(output_path, "t_output_path_family_check_00000_diff_ext");
    std::string result = ascent::expand_path_special_variables(output_file, ".png", -1);
    std::cout << result << std::endl;
    EXPECT_TRUE(result == expected_result);

    remove_test_file(pre_existing_file_name_1);
    remove_test_file(pre_existing_file_name_2);
}
