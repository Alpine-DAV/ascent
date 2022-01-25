//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_query_example1.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <sstream>

#include "ascent.hpp"
#include "conduit_blueprint.hpp"

#include "ascent_tutorial_cpp_utils.hpp"

using namespace ascent;
using namespace conduit;

const int EXAMPLE_MESH_SIDE_DIM = 32;

int main(int argc, char **argv)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);

    // Use Ascent to bin an input mesh in a few ways
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_queries";

    // declare a queries to ask some questions
    Node &queries = add_act["queries"] ;

    // Create a 1D binning projected onto the x-axis
    queries["q1/params/expression"] = "binning('radial','max', [axis('x',num_bins=20)])";
    queries["q1/params/name"] = "1d_binning";

    // Create a 2D binning projected onto the x-y plane
    queries["q2/params/expression"] = "binning('radial','max', [axis('x',num_bins=20), axis('y',num_bins=20)])";
    queries["q2/params/name"] = "2d_binning";

    // Create a binning that emulates a line-out, that is, bin all values
    // between x = [-1,1], y = [-1,1] along the z-axis in 20 bins.
    // The result is a 1x1x20 array
    queries["q3/params/expression"] = "binning('radial','max', [axis('x',[-1,1]), axis('y', [-1,1]), axis('z', num_bins=20)])";
    queries["q3/params/name"] = "3d_binning";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // retrieve the info node that contains the query results
    Node info;
    a.info(info);

    // close ascent
    a.close();

    //
    // We can use the included example python scripts to plot binning results,
    // which are stored in an ascent yaml session file:
    //  plot_binning_1d.py
    //  plot_binning_2d.py
    //  plot_binning_3d.py
    //

    //
    // We can also examine when the results by looking at the expressions
    // results in the output info
    //
    std::cout << info["expressions"].to_yaml() << std::endl;
}


