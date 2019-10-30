#include <iostream>
#include <vector>
#include "conduit.hpp"

using namespace conduit;

int main()
{
    Node n;
    n["dir1/dir2/val1"] = 100.5;
    n.print();
}

