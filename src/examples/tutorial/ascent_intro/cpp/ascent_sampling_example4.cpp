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

const int EXAMPLE_MESH_SIDE_DIM = 20;

void
add_sample_sphere_topology(Node &data,
                           const std::string &topo_name,
                           const std::string &coordset_name,
                           const double radius,
                           const int num_lat,
                           const int num_lon)
{
    // ############################################
    // ### Defining the Topology Coordinate Set ###
    // ############################################

    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> z_vals;
    std::vector<int32> conn;

    const double pi = acos(-1.0);

    //Initialize with north pole values
    x_vals.push_back(0.0);
    y_vals.push_back(0.0);
    z_vals.push_back(radius);

    // Add lat, lon coordinates
    for(int lat = 1; lat < num_lat; ++lat)
    {
        const double theta = pi * static_cast<double>(lat) /
                             static_cast<double>(num_lat);
        const double sin_theta = sin(theta);
        const double cos_theta = cos(theta);

        for(int lon = 0; lon < num_lon; ++lon)
        {
            const double phi = 2.0 * pi * static_cast<double>(lon) /
                               static_cast<double>(num_lon);
            x_vals.push_back(radius * sin_theta * cos(phi));
            y_vals.push_back(radius * sin_theta * sin(phi));
            z_vals.push_back(radius * cos_theta);
        }
    }

    // Add south pole values
    const int32 south_pole = static_cast<int32>(x_vals.size());
    x_vals.push_back(0.0);
    y_vals.push_back(0.0);
    z_vals.push_back(-radius);

    // Add new sphere coordset to mesh
    Node &coords = data["coordsets/" + coordset_name];
    coords["type"] = "explicit";
    coords["values/x"].set(DataType::float64(x_vals.size()));
    coords["values/y"].set(DataType::float64(y_vals.size()));
    coords["values/z"].set(DataType::float64(z_vals.size()));

    float64_array xs = coords["values/x"].value();
    float64_array ys = coords["values/y"].value();
    float64_array zs = coords["values/z"].value();
    for(index_t i = 0; i < static_cast<index_t>(x_vals.size()); ++i)
    {
        xs[i] = x_vals[i];
        ys[i] = y_vals[i];
        zs[i] = z_vals[i];
    }

    // #########################################
    // ### Defining the Topology Conectivity ###
    // #########################################

    auto ring_index = [num_lon](const int lat, const int lon)
    {
        return static_cast<int32>(1 + (lat - 1) * num_lon +
                                  (lon% num_lon));
    };

    for(int lon = 0; lon < num_lon; ++lon)
    {
        conn.push_back(0);
        conn.push_back(ring_index(1, lon + 1));
        conn.push_back(ring_index(1, lon));
    }

    for(int lat = 1; lat < num_lat - 1; ++lat)
    {
        for(int lon = 0; lon < num_lon; ++lon)
        {
            const int32 lower_left = ring_index(lat, lon);
            const int32 lower_right = ring_index(lat, lon + 1);
            const int32 upper_left = ring_index(lat + 1, lon);
            const int32 upper_right = ring_index(lat + 1, lon + 1);

            conn.push_back(lower_left);
            conn.push_back(lower_right);
            conn.push_back(upper_left);

            conn.push_back(lower_right);
            conn.push_back(upper_right);
            conn.push_back(upper_left);
        }
    }

    for(int lon = 0; lon < num_lon; ++lon)
    {
        conn.push_back(ring_index(num_lat - 1, lon));
        conn.push_back(ring_index(num_lat - 1, lon + 1));
        conn.push_back(south_pole);
    }

    Node &topo = data["topologies/" + topo_name];
    topo["type"] = "unstructured";
    topo["coordset"] = coordset_name;
    topo["elements/shape"] = "tri";
    topo["elements/connectivity"].set(DataType::int32(conn.size()));

    int32_array topo_conn = topo["elements/connectivity"].value();
    for(index_t i = 0; i < static_cast<index_t>(conn.size()); ++i)
    {
        topo_conn[i] = conn[i];
    }
}

int main(int argc, char **argv)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);
    add_sample_sphere_topology(mesh, "sample_sphere", "sample_sphere_coords",
                                   10.0, 12, 24);

    // Use Ascent to bin an input mesh in a few ways
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;

    // Add a sampling pipeline
    Node &add_sample_act = actions.append();
    add_sample_act["action"] = "add_pipelines";

    Node &sample_pipe = add_sample_act["pipelines"];
    sample_pipe["pl1/f1/type"] = "sample";
    sample_pipe["pl1/f1/params/fields"] = {"braid"};

    // Define the topology to sample onto
    sample_pipe["pl1/f1/params/topology"] = "sample_sphere";
    sample_pipe["pl1/f1/params/uniform_grid/spacing/dz"] = 1;

    sample_pipe["pl1/f1/params/invalid_value"] = -10.0;

    // Add a scene that renders the sampled result.
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a queries to ask some questions
    Node &scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_name"] = "sample_spherical_topology";

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
    // We can also examine when the results by looking at the expressions
    // results in the output info
    //
    std::cout << info["expressions"].to_yaml() << std::endl;
}