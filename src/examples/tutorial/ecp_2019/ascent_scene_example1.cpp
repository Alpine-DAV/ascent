#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    Ascent a;

    // open ascent
    a.open();

    Node mesh;

    // create the coordinate set
    double X[5] = { -1.0, 0.0, 0.0, 0.0, 1.0 };
    double Y[5] = { 0.0, -1.0, 0.0, 1.0, 0.0 };
    double Z[5] = { 0.0, 0.0, 1.0, 0.0, 0.0 };
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(X, 5);
    mesh["coordsets/coords/values/y"].set_external(Y, 5);
    mesh["coordsets/coords/values/z"].set_external(Z, 5);

    // reference the coordinate set by name
    mesh["topologies/mesh/type"] = "unstructured";
    mesh["topologies/mesh/coordset"] = "coords";
    mesh["topologies/mesh/elements/shape"] = "tet";
    int64 connectivity[8] = { 0, 1, 3, 2, 4, 3, 1, 2 };
    mesh["topologies/mesh/elements/connectivity"].set_external(connectivity, 8);

    const int numCells = 2;
    float vals[numCells] = { 0, 1 };
    float vals2[numCells] = { 1, 0 };
    mesh["fields/variable1/association"] = "element";
    mesh["fields/variable1/topology"] = "mesh";
    mesh["fields/variable1/volume_dependent"] = "false";
    mesh["fields/variable1/values"].set_external(vals, 2);
    mesh["fields/variable2/association"] = "element";
    mesh["fields/variable2/topology"] = "mesh";
    mesh["fields/variable2/volume_dependent"] = "false";
    mesh["fields/variable2/values"].set_external(vals2, 2);

    mesh.print();

    Node verify_info;
    if(!blueprint::mesh::verify(mesh, verify_info))
    {
        std::cout << "Verify failed!" << std::endl;
        verify_info.print();
    }

    // publish mesh to ascent
    a.publish(mesh);

    // declare a scene to render the dataset
    Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "variable1";
    scenes["s1/image_prefix"] = "ascent_output_render_var1";

    scenes["s2/plots/p1/type"] = "pseudocolor";
    scenes["s2/plots/p1/field"] = "variable2";
    scenes["s2/image_prefix"] = "ascent_output_render_var2";

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";
    add_act["scenes"] = scenes;

    // execute
    a.execute(actions);

    // close ascent
    a.close();
}



