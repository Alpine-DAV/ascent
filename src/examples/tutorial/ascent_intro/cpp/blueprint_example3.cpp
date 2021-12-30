//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
    Node &add_act = actions.append();
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
