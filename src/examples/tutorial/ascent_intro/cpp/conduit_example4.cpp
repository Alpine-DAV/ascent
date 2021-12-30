//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_example4.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;
using std::vector;

int main()
{
    //
    // Conduit supports zero copy, allowing a Conduit Node to describe and
    // point to externally allocated data.
    //
    // set_external() is method used to zero copy data into a Node
    //

    Node n;
    vector<int> A1(10);
    A1[0] = 0;
    A1[1] = 1;
    for (int i = 2 ; i < 10 ; i++)
        A1[i] = A1[i-2]+A1[i-1];
    n["fib_deep_copy"].set(A1);

    // create another array to demo difference 
    // between set and set_external
    vector<int> A2(10);

    A2[0] = 0;  
    A2[1] = 1; 
    for (int i = 2 ; i < 10 ; i++)
    {
        A2[i] = A2[i-2]+A2[i-1];
    }

    n["fib_shallow_copy"].set_external(A2);
    A1[9] = -1;
    A2[9] = -1;

    std::cout << n.to_yaml() << std::endl;
}

