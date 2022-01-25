//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_example1.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;

int main()
{
    // 
    // The core of Conduit's data model is `Node` object that
    // holds a dynamic hierarchical key value tree
    //
    // Here is a simple example that creates
    // a key value pair in a Conduit Node:
    //     
    Node n;
    n["key"] = "value";
    std::cout << n.to_yaml() << std::endl;
}

