//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_logger_basic.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent_logging.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
void my_func_nest_2()
{
    ASCENT_MARK_FUNCTION();
    ASCENT_LOG_INFO("nest 2");
}

//-----------------------------------------------------------------------------
void my_func_nest_1()
{
    ASCENT_MARK_FUNCTION();
    ASCENT_LOG_INFO("nest 1");
    my_func_nest_2();
}

//-----------------------------------------------------------------------------
void my_func_nest_0()
{
    ASCENT_MARK_FUNCTION();
    ASCENT_LOG_INFO("nest 0");
    my_func_nest_1();
}

//-----------------------------------------------------------------------------
void myfunc()
{
    ASCENT_MARK_FUNCTION();
    ASCENT_LOG_INFO("I am here!");
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging)
{
    // remove file if it exists
    std::string lfname = "tout_logging_log_1.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    ASCENT_LOG_OPEN(lfname);
    //                                  top level entry count
    ASCENT_LOG_DEBUG("my debug!");
    ASCENT_LOG_INFO("my info!");       // msg (1)
    ASCENT_LOG_WARN("my warning!");    // msg (2)
    ASCENT_MARK_BEGIN("blocky");       // block (3)
        ASCENT_LOG_INFO("my info!");
        ASCENT_LOG_WARN("my warning!");
    ASCENT_MARK_END("blocky");
    myfunc();                          // myfunc (4)

    ASCENT_MARK_BEGIN("blocky");       // block (5)
        myfunc();
    ASCENT_MARK_END("blocky");
    
    ASCENT_MARK_BEGIN("blocky");       // block (6)
       my_func_nest_0();
    ASCENT_MARK_END("blocky");
    ASCENT_LOG_CLOSE();

    conduit::Node n;
    n.load(lfname);
    n.print();
    // 7 messages at root level
    EXPECT_EQ(n.number_of_children(),6);
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_echo)
{
    // remove file if it exists
    std::string lfname = "tout_logging_log_2.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    ascent::Logger::instance().set_echo_threshold(ascent::Logger::LOG_ALL_ID);
    std::cout << "[echoed]" << std::endl;
    ASCENT_LOG_OPEN(lfname);
    ASCENT_LOG_DEBUG("my debug!");
    ASCENT_LOG_INFO("my info!");
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_CLOSE();

    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load(lfname);
    n.print();
    // we should have 2 msgs above debug
    EXPECT_EQ(n.number_of_children(),2); 
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_threshold)
{
    // remove file if it exists
    std::string lfname = "tout_logging_log_3.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    ASCENT_LOG_OPEN(lfname);
    ASCENT_LOG_INFO("my info!");
    ascent::Logger::instance().set_log_threshold(ascent::Logger::LOG_NONE_ID);
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_CLOSE();

    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load(lfname);
    n.print();
    // we should have 1 msg above debug
    EXPECT_EQ(n.number_of_children(),1);
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_error)
{
    bool error_occured = false;
    try
    {
        ASCENT_LOG_ERROR("my error!");
    }
    catch (conduit::Error &error)
    {
        std::cout << error.what() << std::endl;
        error_occured = true;
    }
    EXPECT_TRUE(error_occured);

    // now check error inside log file
    // remove file if it exists
    std::string lfname = "tout_logging_log_4.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    ASCENT_LOG_OPEN(lfname);

    error_occured = false;
    try
    {
        ASCENT_LOG_ERROR("my error!");
    }
    catch (conduit::Error &error)
    {
        std::cout << error.what() << std::endl;
        error_occured = true;
    }
    EXPECT_TRUE(error_occured);

    ASCENT_LOG_CLOSE();
    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load(lfname);
    n.print();
    // we should have 1 msg above debug
    EXPECT_EQ(n.number_of_children(),1);
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_append)
{
    // remove file if it exists
    std::string lfname = "tout_logging_log_5.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    ascent::Logger::instance().set_log_threshold(ascent::Logger::LOG_ALL_ID);
    ASCENT_LOG_OPEN(lfname);
    ASCENT_LOG_INFO("my debug 1!");
    ASCENT_LOG_INFO("my info 1!");
    ASCENT_LOG_WARN("my warning 1!");
    ASCENT_LOG_CLOSE();

    ascent::Logger::instance().set_log_threshold(ascent::Logger::LOG_ALL_ID);
    ASCENT_LOG_OPEN(lfname);
    ASCENT_LOG_INFO("my debug 2!");
    ASCENT_LOG_INFO("my info 2!");
    ASCENT_LOG_WARN("my warning 2!");
    ASCENT_LOG_CLOSE();

    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load(lfname);
    n.print();
    // we should have 5 * 2 (open, close, + 1 debug, + 1 info, + 1 warn)
    EXPECT_EQ(n.number_of_children(),10);
}


//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_error_bad_log_file)
{
    bool error_occured = false;
    try
    {
        ASCENT_LOG_OPEN("/blargh/totally/bogus/des/path/to/log/file.yaml");
    }
    catch (conduit::Error &error)
    {
        std::cout << error.what() << std::endl;
        error_occured = true;
    }
    EXPECT_TRUE(error_occured);
    
    // remove file if it exists
    std::string lfname = "tout_logging_log_6.yaml";
    conduit::utils::remove_path_if_exists(lfname);
    error_occured =false;
    try
    {
        // double open should throw an exception
        ASCENT_LOG_OPEN(lfname);
        ASCENT_LOG_OPEN(lfname);
    }
    catch (conduit::Error &error)
    {
        std::cout << error.what() << std::endl;
        error_occured = true;
    }
    EXPECT_TRUE(error_occured);
    ASCENT_LOG_CLOSE();
}
