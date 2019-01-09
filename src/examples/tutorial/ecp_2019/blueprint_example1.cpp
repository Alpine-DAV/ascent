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

    int numPerDim = 9;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = numPerDim;
    mesh["coordsets/coords/dims/j"] = numPerDim;
    mesh["coordsets/coords/dims/k"] = numPerDim;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = -10.0;
    mesh["coordsets/coords/origin/y"] = -10.0;
    mesh["coordsets/coords/origin/z"] = -10.0;
    double distancePerStep = 20.0/(numPerDim-1);
    mesh["coordsets/coords/spacing/dx"] = distancePerStep;
    mesh["coordsets/coords/spacing/dy"] = distancePerStep;
    mesh["coordsets/coords/spacing/dz"] = distancePerStep;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    int numVertices = numPerDim*numPerDim*numPerDim; // 3D
    float *vals = new float[numVertices];
    for (int i = 0 ; i < numVertices ; i++)
        vals[i] = ( (i%2)==0 ? 0.0 : 1.0);

    mesh["fields/alternating/association"] = "vertex";
    mesh["fields/alternating/topology"] = "topo";
    mesh["fields/alternating/values"].set_external(vals, numVertices);

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
    scenes["s1/plots/p1/field"] = "alternating";
    // Set the output file name (ascent will add ".png")
    scenes["s1/image_prefix"] = "out_ascent_render_uniform";

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";
    add_act["scenes"] = scenes;

    actions.append()["action"] = "execute";

    // execute
    a.execute(actions);

    // close ascent
    a.close();
}



