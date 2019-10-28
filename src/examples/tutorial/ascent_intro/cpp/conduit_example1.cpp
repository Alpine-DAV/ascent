#include <iostream>
#include <vector>
#include "conduit_blueprint.hpp"

using namespace conduit;
using std::vector;

int main()
{
    Node n;
    n["key"] = "value";
    n.print();
}

