//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_tutorial_cpp_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_TUTORIAL_CPP_UTILS_H
#define ASCENT_TUTORIAL_CPP_UTILS_H

#include <iostream>
#include "conduit_blueprint.hpp"

#include <math.h>

using namespace conduit;

const float64 PI_VALUE = 3.14159265359;

// --------------------------------------------------------------------------//
void
tutorial_tets_example(Node &mesh)
{
    mesh.reset();

    //
    // (create example tet mesh from blueprint example 2)
    //
    // Create a 3D mesh defined on an explicit set of points,
    // composed of two tets, with two element associated fields
    //  (`var1` and `var2`)
    //

    // create an explicit coordinate set
    double X[5] = { -1.0, 0.0, 0.0, 0.0, 1.0 };
    double Y[5] = { 0.0, -1.0, 0.0, 1.0, 0.0 };
    double Z[5] = { 0.0, 0.0, 1.0, 0.0, 0.0 };
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set(X, 5);
    mesh["coordsets/coords/values/y"].set(Y, 5);
    mesh["coordsets/coords/values/z"].set(Z, 5);


    // add an unstructured topology
    mesh["topologies/mesh/type"] = "unstructured";
    // reference the coordinate set by name
    mesh["topologies/mesh/coordset"] = "coords";
    // set topology shape type
    mesh["topologies/mesh/elements/shape"] = "tet";
    // add a connectivity array for the tets
    int64 connectivity[8] = { 0, 1, 3, 2, 4, 3, 1, 2 };
    mesh["topologies/mesh/elements/connectivity"].set(connectivity, 8);

    const int num_elements = 2;
    float var1_vals[num_elements] = { 0, 1 };
    float var2_vals[num_elements] = { 1, 0 };
    
    // create a field named var1
    mesh["fields/var1/association"] = "element";
    mesh["fields/var1/topology"] = "mesh";
    mesh["fields/var1/values"].set(var1_vals, 2);

    // create a field named var2
    mesh["fields/var2/association"] = "element";
    mesh["fields/var2/topology"] = "mesh";
    mesh["fields/var2/values"].set(var2_vals, 2);

    //  make sure the mesh we created conforms to the blueprint
    Node verify_info;
    if(!blueprint::mesh::verify(mesh, verify_info))
    {
        std::cout << "Mesh Verify failed!" << std::endl;
        std::cout << verify_info.to_yaml() << std::endl;
    }
}

// --------------------------------------------------------------------------//
void
tutorial_gyre_example(float64 time_value, Node &mesh)
{
    mesh.reset();
    int xy_dims = 40;
    int z_dims = 2;
    
    conduit::blueprint::mesh::examples::braid("hexs",
                                             xy_dims,
                                             xy_dims,
                                             z_dims,
                                             mesh);

    mesh["state/time"] = time_value;
    Node &field = mesh["fields/gyre"];
    field["association"] = "vertex";
    field["topology"] = "mesh";
    field["values"].set(DataType::float64(xy_dims*xy_dims*z_dims));
    
    Node &vec_field = mesh["fields/gyre_vel"];
    vec_field["association"] = "vertex";
    vec_field["topology"] = "mesh";
    vec_field["values/u"].set(DataType::float64(xy_dims*xy_dims*z_dims));
    vec_field["values/v"].set(DataType::float64(xy_dims*xy_dims*z_dims));
    vec_field["values/w"].set(DataType::float64(xy_dims*xy_dims*z_dims));

    float64 *values_ptr = field["values"].value();
    float64 *u_values_ptr = vec_field["values/u"].value();
    float64 *v_values_ptr = vec_field["values/v"].value();
    float64 *w_values_ptr = vec_field["values/w"].value();

    float64 e = 0.25;
    float64 A = 0.1;
    float64 w = (2.0 * PI_VALUE) / 10.0;
    float64 a_t = e * sin(w * time_value);
    float64 b_t = 1.0 - 2 * e * sin(w * time_value);
    // print("e: " + str(e) + " A " + str(A) + " w " + str(w) + " a_t " + str(a_t) + " b_t " + str(b_t))
    // print(b_t)
    // print(w)
    int idx = 0;
    for (int z=0; z < z_dims; z++)
    {
        for (int y=0; y < xy_dims; y++)
        {
            // scale y to 0-1
            float64 y_n = float64(y)/float64(xy_dims);
            float64 y_t = sin(PI_VALUE * y_n);
            for (int x=0; x < xy_dims; x++)
            {
                // scale x to 0-1
                float64 x_f = float(x)/ (float(xy_dims) * .5);
                float64 f_t = a_t * x_f * x_f + b_t * x_f;
                // print(f_t)
                float64 value = A * sin(PI_VALUE * f_t) * y_t;
                float64 u = -PI_VALUE * A * sin(PI_VALUE * f_t) * cos(PI_VALUE * y_n);
                float64 df_dx = 2.0 * a_t + b_t;
                // print("df_dx " + str(df_dx))
                float64 v = PI_VALUE * A * cos(PI_VALUE * f_t) * sin(PI_VALUE * y_n) * df_dx;
                values_ptr[idx] = sqrt(u * u + v * v);
                u_values_ptr[idx] = u;
                v_values_ptr[idx] = v;
                w_values_ptr[idx] = 0;
                // values[idx] = u * u + v * v
                // values[idx] = value
                // print("u " + str(u) + " v " + str(v) + " mag " + str(math.sqrt(u * u + v * v)))
                idx++;
            }
        }
    }

    //print(values)
}

#endif
