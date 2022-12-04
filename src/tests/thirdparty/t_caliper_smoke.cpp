// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_caliper_smoke.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

#include <iostream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
void
test_print(int j)
{
    CALI_CXX_MARK_FUNCTION;
    std::cout << "Here we go: " << j << std::endl;
}


//-----------------------------------------------------------------------------
void
test_it(int i)
{
    CALI_CXX_MARK_FUNCTION;
    std::cout << "it is: " << i << std::endl;
    for(int j=0;j<5;j++)
    {
        test_print(j);
    }
}


//-----------------------------------------------------------------------------
TEST(caliper_smoke, basic_use)
{

    cali::ConfigManager mgr;
    mgr.add("runtime-report");
    mgr.start();
    for(int i=0;i<2;i++)
    {
        test_it(i);
    }
    mgr.flush();
}
