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
void myfunc()
{
    ASCENT_MARK_FUNCTION();
    ASCENT_LOG_INFO("I am here!");
}

//-----------------------------------------------------------------------------
TEST(ascent_smoke, ascent_about)
{
    ASCENT_LOG_OPEN("here_we_go.yaml")
    ASCENT_LOG_INFO("my info!");
    ASCENT_LOG_WARN("my warning!");
    ASCENT_LOG_ERROR("my error!");
    ASCENT_MARK_BEGIN("blocky");
        ASCENT_LOG_INFO("my info!");
        ASCENT_LOG_WARN("my warning!");
        ASCENT_LOG_ERROR("my error!");
    ASCENT_MARK_END("blocky");
    myfunc();

    ASCENT_MARK_BEGIN("blocky2");
        myfunc();
    ASCENT_MARK_END("blocky2");
}

