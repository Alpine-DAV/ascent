#include <iostream>
#include <vector>
#include "conduit_blueprint.hpp"

using namespace conduit;
using std::vector;

int main()
{
    Node n;
    n["dir1/dir2/val1"] = 100.5;
    n.print();
}

