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

int main(int argc, char **argv)
{
    // Use Ascent to extract mesh cycle and entropy of a time varying mesh
    Ascent a;

    // open ascent
    a.open();

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_queries";

    // declare a queries to ask some questions
    Node &queries = add_act["queries"] ;

    // add a simple query expression (q1)
    queries["q1/params/expression"] = "cycle()";;
    queries["q1/params/name"] = "cycle";

    // add a more complex query expression (q2)
    queries["q2/params/expression"] = "entropy(histogram(field('gyre'), num_bins=128))";
    queries["q2/params/name"] = "entropy_of_gyre";

    // declare a scene to render the dataset
    Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "gyre";
    // Set the output file name (ascent will add ".png")
    scenes["s1/image_name"] = "out_gyre";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // gyre time varying params
    int nsteps = 10;
    float time_value = 0.0;
    float delta_time = 0.5;

    Node mesh;
    Node info;

    for( int step =0; step < nsteps; step++)
    {
        // call helper that generates a gyre time varying example mesh.
        // gyre ref :https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
        tutorial_gyre_example(time_value, mesh);

        // update the example cycle
        int cycle = 100 + step * 100;
        mesh["state/cycle"] = cycle;
        std::cout << "time: " << time_value << " cycle: " << cycle << std::endl;

        // publish mesh to ascent
        a.publish(mesh);

        // update image name
        std::ostringstream oss;
        oss << "out_gyre_" << std::setfill('0') << std::setw(4) << step;
        scenes["s1/image_name"] = oss.str();

        // execute the actions
        a.execute(actions);

        // retrieve the info node that contains the query results
        Node ts_info;
        a.info(ts_info);

        // add to our running info
        info["expressions"].update(ts_info["expressions"]);

        // update time
        time_value = time_value + delta_time;
    }

    // close ascent
    a.close();

    // view the results of the cycle query
    std::cout << info["expressions/cycle"].to_yaml() << std::endl;
    // Note that query results can be indexed by cycle

    // view the results of the cycle query
    std::cout << info["expressions/entropy_of_gyre"].to_yaml() << std::endl;

    // create an array with the entropy values from all 
    // cycles
    Node entropy;
    entropy.set(DataType::float64(nsteps));
    float64 *entropy_vals_ptr = entropy.value();

    // get the node that has the time history
    Node &gyre = info["expressions/entropy_of_gyre"];

    // reformat conduit data into  summary array
    for(int i=i; i < gyre.number_of_children(); i++ )
    {
        entropy_vals_ptr[i] = gyre[i]["value"].to_float64();
    }

    std::cout << "Entropy Result" << std::endl;
    std::cout << entropy.to_yaml() << std::endl;
}


