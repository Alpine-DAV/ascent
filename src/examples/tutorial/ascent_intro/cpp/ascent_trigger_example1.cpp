//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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
    // Use triggers to render when conditions occur
    Ascent a;

    // open ascent
    a.open();

    // setup actions
    Node actions;

    // declare a question to ask 
    Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";

    // add our entropy query (q1)
    Node &queries = add_queries["queries"];
    queries["q1/params/expression"] = "entropy(histogram(field('gyre'), num_bins=128))";
    queries["q1/params/name"] = "entropy";

    // declare triggers 
    Node &add_triggers = actions.append();
    add_triggers["action"] = "add_triggers";
    Node &triggers = add_triggers["triggers"];

    // add a simple trigger (t1_ that fires at cycle 500
    triggers["t1/params/condition"] = "cycle() == 500";
    triggers["t1/params/actions_file"] = "cycle_trigger_actions.yaml";

    // add trigger (t2) that fires when the change in entroy exceeds 0.5

    // the history function allows you to access query results of previous
    // cycles. relative_index indicates how far back in history to look.

    // Looking at the plot of gyre entropy in the previous notebook, we see a jump
    // in entropy at cycle 200, so we expect the trigger to fire at cycle 200
    triggers["t2/params/condition"] = "entropy - history(entropy, relative_index = 1) > 0.5";
    triggers["t2/params/actions_file"] = "entropy_trigger_actions.yaml";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // gyre time varying params
    int nsteps = 10;
    float time_value = 0.0;
    float delta_time = 0.5;
    
    Node mesh;

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

        // execute the actions
        a.execute(actions);

        // update time
        time_value = time_value + delta_time;
    }
    // retrieve the info node that contains the query results
    Node info;
    a.info(info);

    // close ascent
    a.close();

    // this will render:
    //   cycle_trigger_out_500.png
    //   entropy_trigger_out_200.png
    //
    //
    // We can also examine when the triggers executed by looking at the expressions
    // results in the output info
    //

    std::cout << info["expressions"].to_yaml() << std::endl;
}


