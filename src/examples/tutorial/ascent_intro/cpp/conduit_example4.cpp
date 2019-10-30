#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;
using std::vector;

int main()
{
    Node n;
    vector<int> A1(20);
    A1[0] = 0;
    A1[1] = 1;
    for (int i = 2 ; i < 20 ; i++)
        A1[i] = A1[i-2]+A1[i-1];
    n["fib_deep_copy"].set(A);
    n.print();
    
    // create another array to demo diff
    // between set and set_external
    vector<int> A2(20);
    
    A2[0] = 0;  
    A2[1] = 1; 
    for (int i = 2 ; i < 20 ; i++)
    {
        A2[i] = A2[i-2]+A2[i-1];
    }

    n["fib_shallow_copy"].set_external(A2);
    A1[10] = -1;
    A2[10] = -1;
    n.print();
}

