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
/// file: blueprint_example3.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <math.h>
#include <sstream>

#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

const float64 PI_VALUE = 3.14159265359;

// The conduit blueprint library provides several
// simple builtin examples that cover the range of
// supported coordinate sets, topologies, field etc
//
// Here we create a mesh using the braid example
// (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid)
// and modify one of its fields to create a time-varying
// example

// Define a function that will calcualte a time varying field
void braid_time_varying(int npts_x,
                        int npts_y,
                        int npts_z,
                        float interp,
                        Node & res)
{
    if(npts_z < 1)
        npts_z = 1;

    int npts = npts_x * npts_y * npts_z;

    res["association"] = "vertex";
    res["topology"]    = "mesh";

    float64 *vals_ptr = res["values"].value();

    float64 dx_seed_start = 0.0;
    float64 dx_seed_end   = 5.0;
    float64 dx_seed = interp * (dx_seed_end - dx_seed_start) + dx_seed_start;

    float64 dy_seed_start = 0.0;
    float64 dy_seed_end   = 2.0;
    float64 dy_seed = interp * (dy_seed_end - dy_seed_start) + dy_seed_start;

    float64 dz_seed = 3.0;

    float64 dx = (float64) (dx_seed * PI_VALUE) / float64(npts_x - 1);
    float64 dy = (float64) (dy_seed * PI_VALUE) / float64(npts_y-1);
    float64 dz = (float64) (dz_seed * PI_VALUE) / float64(npts_z-1);

    int idx = 0;
    for (int k=0; k < npts_z; k++)
    {
        float64 cz =  (k * dz) - (1.5 * PI_VALUE);
        for (int j=0; j < npts_y; j++)
        {
            float64 cy =  (j * dy) - PI_VALUE;
            for (int i=0; i < npts_x; i++)
            {
                float64 cx = (i * dx) + (2.0 * PI_VALUE);
                float64 cv = sin( cx ) + 
                             sin( cy ) +  
                             2.0 * cos(sqrt( (cx*cx)/2.0 +cy*cy) / .75) + 
                             4.0 * cos( cx*cy / 4.0);
                                  
                if(npts_z > 1)
                {
                    cv += sin( cz ) + 
                          1.5 * cos(sqrt(cx*cx + cy*cy + cz*cz) / .75);
                }
                vals_ptr[idx] = cv;
                idx++;
            }
        }
    }
}

int main(int argc, char **argv)
{
    // create a conduit node with an example mesh using conduit blueprint's braid function
    // ref: https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid

    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              50,
                                              50,
                                              50,
                                              mesh);

    Ascent a;
    // open ascent
    a.open();

    // create our actions
    Node actions;
    Node & add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a scene (s1) and plot (p1)
    // to render braid field
    Node & scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    // print our actions tree
    std::cout << actions.to_yaml() << std::endl;

    // loop over a set of steps and
    // render a time varying version of the braid field

    int nsteps = 20;

    float64 interp_val = 0.0;
    float64 interp_dt = 1.0 / float64(nsteps-1);

    for( int i=0; i < nsteps; i++)
    {
        std::cout << i << ": interp = " << interp_val << std::endl;

        // update the braid field
        braid_time_varying(50,50,50,interp_val,mesh["fields/braid"]);
        // update the mesh cycle
        mesh["state/cycle"] = i;
        // Set the output file name (ascent will add ".png")
        std::ostringstream oss;
        oss << "out_ascent_render_braid_tv_" << i;
        scenes["s1/renders/r1/image_name"] = oss.str();
        scenes["s1/renders/r1/camera/azimuth"] = 25.0;

        // publish mesh to ascent
        a.publish(mesh);

        // execute the actions
        a.execute(actions);

        interp_val += interp_dt;
    }

    // close ascent
    a.close();
}
