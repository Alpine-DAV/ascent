//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: blueprint_example2.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;


int main(int argc, char **argv)
{
    //
    // Create a 3D mesh defined on an explicit set of points,
    // composed of two tets, with two element associated fields
    //  (`var1` and `var2`)
    //

    Node mesh;

    // create an explicit coordinate set
    double X[5] = { -1.0, 0.0, 0.0, 0.0, 1.0 };
    double Y[5] = { 0.0, -1.0, 0.0, 1.0, 0.0 };
    double Z[5] = { 0.0, 0.0, 1.0, 0.0, 0.0 };
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(X, 5);
    mesh["coordsets/coords/values/y"].set_external(Y, 5);
    mesh["coordsets/coords/values/z"].set_external(Z, 5);


    // add an unstructured topology
    mesh["topologies/mesh/type"] = "unstructured";
    // reference the coordinate set by name
    mesh["topologies/mesh/coordset"] = "coords";
    // set topology shape type
    mesh["topologies/mesh/elements/shape"] = "tet";
    // add a connectivity array for the tets
    int64 connectivity[8] = { 0, 1, 3, 2, 4, 3, 1, 2 };
    mesh["topologies/mesh/elements/connectivity"].set_external(connectivity, 8);

    const int num_elements = 2;
    float var1_vals[num_elements] = { 0, 1 };
    float var2_vals[num_elements] = { 1, 0 };
    
    // create a field named var1
    mesh["fields/var1/association"] = "element";
    mesh["fields/var1/topology"] = "mesh";
    mesh["fields/var1/values"].set_external(var1_vals, 2);

    // create a field named var2
    mesh["fields/var2/association"] = "element";
    mesh["fields/var2/topology"] = "mesh";
    mesh["fields/var2/values"].set_external(var2_vals, 2);

    // print the mesh we created
    std::cout << mesh.to_yaml() << std::endl;

    //  make sure the mesh we created conforms to the blueprint

    Node verify_info;
    if(!blueprint::mesh::verify(mesh, verify_info))
    {
        std::cout << "Mesh Verify failed!" << std::endl;
        std::cout << verify_info.to_yaml() << std::endl;
    }
    else
    {
        std::cout << "Mesh verify success!" << std::endl;
    }

    // now lets look at the mesh with Ascent
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node & add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a scene (s1) with one plot (p1) 
    // to render the dataset
    Node &scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "var1";
    // Set the output file name (ascent will add ".png")
    scenes["s1/image_name"] = "out_ascent_render_tets";

    // print our full actions tree
    std::cout <<  actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}

