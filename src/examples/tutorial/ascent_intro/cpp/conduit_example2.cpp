//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_example2.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;

int main()
{
    // 
    // Using hierarchical paths imposes a tree structure
    // 
    Node n;
    n["dir1/dir2/val1"] = 100.5;
    std::cout << n.to_yaml() << std::endl;
}

