//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_3d.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;
#include <conduit_fmt/conduit_fmt.h>

index_t EXAMPLE_MESH_SIDE_DIM = 20;
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("hexs",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node pipelines;
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "project_2d";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["image_width"] = 512;
//     params["image_height"] = 512;
//
//     conduit::Node extracts;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     conduit::Node actions;
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     add_extracts["extracts"] = extracts;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_contour)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("hexs",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering of a Contour");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_contour");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node pipelines;
//
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "contour";
//     // filter knobs
//     conduit::Node &contour_params = pipelines["pl1/f1/params"];
//     contour_params["field"] = "braid";
//     contour_params["iso_values"] = 0.;
//
//     pipelines["pl1/f2/type"] = "project_2d";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f2/params"];
//     params["image_width"] = 512;
//     params["image_height"] = 512;
//
//     conduit::Node extracts;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     conduit::Node actions;
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     add_extracts["extracts"] = extracts;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering of a contour";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_fields_specified)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("hexs",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering with fields specified");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_fields_specified");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node pipelines;
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "project_2d";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["image_width"] = 512;
//     params["image_height"] = 512;
//     params["fields"].append() = "braid";
//
//     conduit::Node extracts;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     conduit::Node actions;
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     add_extracts["extracts"] = extracts;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering of specific fields";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_data_bounds_specified)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("quads",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               0,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering with fields specified");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_bounds_specified");
//
//     conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
//                                                 "tout_scalar_rendering_bounds_specified_input"),"hdf5");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node pipelines;
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "project_2d";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["image_width"] = 512;
//     params["image_height"] = 512;
//     params["dataset_bounds"] = {  0.0, 5.0, // x
//                                 -10.0,-5.0, // y
//                                   0.0,0.0}; // z
//
//     conduit::Node extracts;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     conduit::Node actions;
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     add_extracts["extracts"] = extracts;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering of specific fields";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_2d_camera)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("quads",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               0,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering with a 2d camera");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_2d_camera");
//
//     conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
//                                                   "tout_scalar_rendering_2d_camera_input"),"hdf5");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node actions;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     conduit::Node &pipelines = add_pipelines["pipelines"];
//     pipelines["pl1/f1/type"] = "project_2d";
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["image_width"]  = 512;
//     params["image_height"] = 512;
//     params["camera/2d"] = { -7.0, 3.0, 0.0,4.0 };
//
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     conduit::Node &extracts=add_extracts["extracts"];;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering with 2d camera mode";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_field_filtering)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("quads",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               0,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering with a 2d camera");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_2d_field_filtering");
//
//     conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
//                                                   "tout_scalar_rendering_2d_field_filtering_input"),"hdf5");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node actions;
//     conduit::Node &declare_fields= actions.append();
//     declare_fields["action"] = "declare_fields";
//     declare_fields["fields"].append() = "braid";
//     declare_fields["fields"].append() = "radial";
//
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     conduit::Node &pipelines = add_pipelines["pipelines"];
//     pipelines["pl1/f1/type"] = "project_2d";
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["image_width"]  = 512;
//     params["image_height"] = 512;
//     params["camera/2d"] = { -7.0, 3.0, 0.0,4.0 };
//
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     conduit::Node &extracts=add_extracts["extracts"];;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//
//     Ascent ascent;
//     conduit::Node opts;
//     opts["field_filtering"] = "true";
//     ascent.open(opts);
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering with filter filtering";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
// //-----------------------------------------------------------------------------
// TEST(ascent_scalar_rendering, test_scalar_rendering_contour_rays_output)
// {
//     // the viskores runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with viskores support
//     if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent support disabled, skipping test");
//         return;
//     }
//
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("hexs",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     ASCENT_INFO("Testing Scalar Rendering of a Contour with Rays Output");
//
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_contour_rays");
//
//     //
//     // Create the actions.
//     //
//
//     conduit::Node pipelines;
//
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "contour";
//     // filter knobs
//     conduit::Node &contour_params = pipelines["pl1/f1/params"];
//     contour_params["field"] = "braid";
//     contour_params["iso_values"] = 0.;
//
//     pipelines["pl1/f2/type"] = "project_2d";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f2/params"];
//     params["image_width"] = 512;
//     params["image_height"] = 512;
//     params["result"] = "rays";
//
//     conduit::Node extracts;
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"] = "pl1";
//
//     extracts["e1/params/path"] = output_file;
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     conduit::Node actions;
//     // add the extracts
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     add_extracts["extracts"] = extracts;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     std::string msg = "An example of scalar rendering of a contour";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
void gen_rays_mesh(const Node &params,
                   const std::string &ofname)
{
    index_t nrays = (index_t) params["rays/points"].dtype().number_of_elements() / 3;
    index_t npts = nrays *2;
    float64 max_dist = params["rays/max_distance"].value();

    float64_accessor pts   = params["rays/points"].value();
    float64_accessor norms = params["rays/normals"].value();


    Node mesh;

    mesh["coordsets/pts/type"] = "explicit";
    mesh["coordsets/pts/values/x"] = DataType::float64(npts);
    mesh["coordsets/pts/values/y"] = DataType::float64(npts);
    mesh["coordsets/pts/values/z"] = DataType::float64(npts);

    mesh["topologies/rays/type"] = "unstructured";
    mesh["topologies/rays/coordset"] = "pts";
    mesh["topologies/rays/connectivity"] = "pts";
    mesh["topologies/rays/elements/shape"] = "line";
    mesh["topologies/rays/elements/connectivity"].set(DataType::index_t(npts));
    index_t_array ray_conn = mesh["topologies/rays/elements/connectivity"].value();
    float64_array vals_x = mesh["coordsets/pts/values/x"].value();
    float64_array vals_y = mesh["coordsets/pts/values/y"].value();
    float64_array vals_z = mesh["coordsets/pts/values/z"].value();

    index_t pts_idx =0;
    index_t idx =0;
    for(index_t i=0;i<nrays;i++)
    {
        vals_x[idx] = pts[pts_idx];
        vals_x[idx+1] = pts[pts_idx] + norms[pts_idx] * max_dist;

        vals_y[idx] = pts[pts_idx+1];
        vals_y[idx+1] = pts[pts_idx+1] + norms[pts_idx+1] * max_dist;

        vals_z[idx] = pts[pts_idx+2];
        vals_z[idx+1] = pts[pts_idx+2] + norms[pts_idx+2] * max_dist;
        ray_conn[idx] = idx;
        ray_conn[idx+1] = idx+1;

        idx+=2;
        pts_idx+=3;

    }

    mesh["fields/id/association"] = "element";
    mesh["fields/id/topology"] = "rays";
    mesh["fields/id/values"] = DataType::float64(nrays);
    
    float64_array fv = mesh["fields/id/values"].value();
    for(index_t i=0;i<nrays;i++)
    {
        fv[i] = i;
    }

    std::cout << "full: " << mesh.to_yaml() << std::endl;

    conduit::Node info;
    if(!conduit::blueprint::mesh::verify(mesh,info))
    {
        std::cout << info.to_yaml() << std::endl;
    }
    else
    {

    static int rr = 0;

    conduit::relay::io::blueprint::save_mesh(mesh,conduit_fmt::format("tout_rr_{:06d}",rr));

    rr++;

    }

}

//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_contour_explicit_rays_with_rays_output)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Scalar Rendering of a Contour Explicit Rays with Rays Output");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_explicit_rays");

    //
    // Create the actions.
    //

    conduit::Node pipelines;

    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 1.5;

    pipelines["pl2/f2/type"] = "project_2d";
    pipelines["pl2/pipeline"] = "pl1";
    
    // filter knobs
    conduit::Node &params = pipelines["pl2/f2/params"];

    // params["rays/points"]  = { 2.5, 0.0, 0.0};
    //
    // params["rays/normals"] = {0.0, 0.0, 1.0};



    params["rays/points"]  = { 0.0, 0.0, 0.0, // center
                              -2.5, 0.0, 0.0, // left
                               2.5, 0.0, 0.0, // right
                               0.0, -2.5, 0.0,// bottom
                               0.0, 2.5, 0.0, // top
                               //
                               20.0, 20.0, 5.0, // top
                               //
                               0.0, 20.0, -5.0,
                               0.0, 20.0, 5.0,
                               0.0, 20.0, 0.0, // x2
                               
                               20.0, -5.0, 0.0,
                               20.0, 5.0, 0.0,
                               20.0, 0.0, 0.0}; // x2
                               // 15.0, 15.0, 0.0};  // top

    params["rays/normals"] = {0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              //
                              -0.7071, -0.7071, 0.0,

                              //
                              0.0, -1.0, 0.0,
                              0.0, -1.0, 0.0,
                              0.0, -1.0, 0.0,
                              //
                              -1.0, 0.0, 0.0,
                              -1.0, 0.0, 0.0,
                              -1.0, 0.0, 0.0};

    params["rays/max_distance"] = 20.0;
    params["result"] = "rays";
    gen_rays_mesh(params,"here");
    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file + "_input";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    extracts["e2/type"]  = "relay";
    extracts["e2/pipeline"] = "pl2";

    extracts["e2/params/path"] = output_file + "_res";
    extracts["e2/params/protocol"] = "blueprint/mesh/hdf5";


    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of a contour";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}


