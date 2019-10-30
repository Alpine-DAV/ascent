#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;

int main()
{
    Node n;
    int *A = new int[20];
    A[0] = 0;
    A[1] = 1;
    for (int i = 2 ; i < 20 ; i++)
        A[i] = A[i-2]+A[i-1];
    n["fib"].set(A, 20);

    n.print();

    delete [] A;
}

