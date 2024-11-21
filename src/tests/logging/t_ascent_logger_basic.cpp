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

    ASCENT_LOG_OPEN("tout_logging_log_1.yaml");
    ASCENT_LOG_INFO("my info!");
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_ERROR("my error!");
    ASCENT_MARK_BEGIN("blocky");
        ASCENT_LOG_INFO("my info!");
        ASCENT_LOG_WARN("my warning!");
        ASCENT_LOG_ERROR("my error!");
    ASCENT_MARK_END("blocky");
    myfunc();

    ASCENT_MARK_BEGIN("blocky");
        myfunc();
    ASCENT_MARK_END("blocky");
    
    ASCENT_MARK_BEGIN("blocky");
       my_func_nest_0();
    ASCENT_MARK_END("blocky");
    ASCENT_LOG_CLOSE();

    conduit::Node n;
    n.load("tout_logging_log_1.yaml");
    n.print();
    
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_echo)
{

    std::cout << "[echoed]" << std::endl;
    ASCENT_LOG_OPEN("tout_logging_log_2.yaml");
    ascent::Logger::instance()->set_echo_threshold(0);
    ASCENT_LOG_INFO("my info!");
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_ERROR("my error!");
    ASCENT_LOG_CLOSE();

    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load("tout_logging_log_2.yaml");
    n.print();
    EXPECT_EQ(n.number_of_children(),3);
}

//-----------------------------------------------------------------------------
TEST(ascent_logging, basic_logging_threshold)
{

    ASCENT_LOG_OPEN("tout_logging_log_3.yaml");
    ASCENT_LOG_INFO("my info!");
    ascent::Logger::instance()->set_log_threshold(ascent::Logger::LEGENDARY);
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_ERROR("my error!");
    ASCENT_LOG_CLOSE();

    std::cout << "[loaded]" << std::endl;
    conduit::Node n;
    n.load("tout_logging_log_3.yaml");
    n.print();
    EXPECT_EQ(n.number_of_children(),1);
}



