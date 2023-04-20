//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_umpire_smoke.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"
#include <umpire/Umpire.hpp>
#include <iostream>

//-----------------------------------------------------------------------------
TEST(umpire_smoke, umpire_host_allocate)
{
    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator host_allocator = rm.getAllocator("HOST");
    void *ptr = host_allocator.allocate(4);
    host_allocator.deallocate(ptr);
}

//-----------------------------------------------------------------------------
TEST(umpire_smoke, umpire_device_allocate)
{
#if defined(UMPIRE_ENABLE_DEVICE)
    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
    void *ptr = device_allocator.allocate(4);
    device_allocator.deallocate(ptr);
#else
    std::cout << "Skipping device allocated test (Umpire built w/o device support)" << std::endl;
#endif
}

