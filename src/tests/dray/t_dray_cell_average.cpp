// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <dray/filters/cell_average.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
bool
mfem_enabled()
{
#ifdef DRAY_MFEM_ENABLED
    return true;
#else
    return false;
#endif
}

TEST(dray_cell_average, smoke)
{
    std::cout << "Hello, world!" << std::endl;
    dray::CellAverage filter;
    (void)filter;
}

TEST(dray_cell_average, normal_cube)
{
    
}