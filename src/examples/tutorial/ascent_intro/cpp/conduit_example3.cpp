//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_example3.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;

int main()
{
    //
    // Conduit's Node trees hold strings or bitwidth style numeric array leaves
    //
    // In C++: you can pass raw pointers to numeric arrays or
    // std::vectors with numeric values
    //
    // In python: numpy ndarrays are used for arrays
    //

    Node n;
    int *A = new int[10];
    A[0] = 0;
    A[1] = 1;
    for (int i = 2 ; i < 10 ; i++)
        A[i] = A[i-2]+A[i-1];
    n["fib"].set(A, 10);

    std::cout << n.to_yaml() << std::endl;

    delete [] A;
}

