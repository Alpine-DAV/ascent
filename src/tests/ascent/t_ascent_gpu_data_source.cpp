//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_gpu_data_source.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_annotations.hpp>

#include <iostream>
#include <cstring>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

#if defined(ASCENT_HIP_ENABLED)
#include <hip/hip_runtime.h>
#endif

using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
// memory helpers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void *
device_alloc(int size)
{
#if defined (ASCENT_CUDA_ENABLED)
    void *buff;
    cudaMalloc(&buff, size);
    return buff;
#elif defined (ASCENT_HIP_ENABLED)
    void *buff;
    hipMalloc(&buff, size);
    return buff;
#else
    return malloc(size);
#endif
}

//-----------------------------------------------------------------------------
void
device_free(void *ptr)

{
#if defined (ASCENT_CUDA_ENABLED)
    cudaFree(ptr);
#elif defined (ASCENT_HIP_ENABLED)
    hipFree(ptr);
#else
    free(ptr);
#endif
}

//-----------------------------------------------------------------------------
void
copy_from_device_to_host(void *dest, void *src, int size)
{
#if defined (ASCENT_CUDA_ENABLED)
   cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
#elif defined (ASCENT_HIP_ENABLED)
   hipMemcpy(dest, src, size, hipMemcpyDeviceToHost);
#else
   memcpy(dest,src,size);
#endif
}


//-----------------------------------------------------------------------------
void
copy_from_host_to_device(void *dest, void *src, int size)
{
#if defined (ASCENT_CUDA_ENABLED)
   cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
#elif defined (ASCENT_HIP_ENABLED)
   hipMemcpy(dest, src, size, hipMemcpyHostToDevice);
#else
   memcpy(dest,src,size);
#endif
}

//-----------------------------------------------------------------------------
void
device_move(conduit::Node &data)
{
    // alloc proper size
    index_t data_nbytes = data.total_bytes_allocated();
    void *device_ptr = device_alloc(data_nbytes);
    copy_from_host_to_device(device_ptr,data.data_ptr(),data_nbytes);
    conduit::DataType dtype = data.dtype();
    data.set_external(dtype,device_ptr);
}

//-----------------------------------------------------------------------------
void
device_cleanup(conduit::Node &data)
{
    void *device_ptr = data.data_ptr();
    device_free(device_ptr);
    data.reset();
}


index_t EXAMPLE_MESH_SIDE_DIM = 20;
//-----------------------------------------------------------------------------
TEST(ascent_gpu_data_source, test_gpu_source_contour_and_render_3d)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D on device tests");
        return;
    }

    string output_path = prepare_output_dir();
    std::string tout_annot_file = conduit::utils::join_file_path(output_path,
                                                                 "tout_render_gpu_source_data_annotations_file.txt");
    // clean up output file if it exists
    conduit::utils::remove_path_if_exists(tout_annot_file);

#if defined (ASCENT_CUDA_ENABLED)
    std::cout << "[using ascent cuda support]" << std::endl;
#elif defined (ASCENT_HIP_ENABLED)
    std::cout << "[using ascent hip support]" << std::endl;
#else
    std::cout << "[WARNING: ascent lacks device (cuda, hip) support]" << std::endl;
#endif
  

    if(ascent::annotations::supported())
    {
        std::cout << "[ascent annotations enabled]" << std::endl;
    }
    else
    {
        std::cout << "[WARNING: ascent lacks annotations (caliper) support]" << std::endl;
    }

    conduit::Node opts;
    opts["config"] = "runtime-report";
    opts["output_file"] = tout_annot_file;



    ascent::annotations::initialize(opts);

    ASCENT_ANNOTATE_MARK_BEGIN("host_create_data");
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

    ASCENT_ANNOTATE_MARK_END("host_create_data");


    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_gpu_source_data_1_");

    // remove old images before rendering
    remove_test_image(output_file);

    ASCENT_ANNOTATE_MARK_BEGIN("ascent_setup_actions");
    //
    // Create the actions.
    //
    conduit::Node actions;

    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 0.0;

    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/path"] = conduit::utils::join_file_path(output_path,
                                                        "tout_ext_gpu_source_data_1");
    

    conduit::Node &sinfo = actions.append();
    sinfo["action"] = "save_info";
    sinfo["file_name"] = conduit::utils::join_file_path(output_path,
                                                        "tout_info_gpu_source_data_1.yaml");

    
    
    ASCENT_ANNOTATE_MARK_END("ascent_setup_actions");
    //
    // Run Ascent
    //
    Ascent ascent;
    ASCENT_ANNOTATE_MARK_BEGIN("ascent_copy_and_run_on_device");
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    ASCENT_ANNOTATE_MARK_END("ascent_copy_and_run_on_device");

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));

    ASCENT_ANNOTATE_MARK_BEGIN("host_copy_to_device");
    //now move coords, connectivity, and fields to the device

    // device_move allocates and uses set external to provide
    // new device data
    device_move(data["coordsets/coords/values/x"]);
    device_move(data["coordsets/coords/values/y"]);
    device_move(data["coordsets/coords/values/z"]);
    device_move(data["topologies/mesh/elements/connectivity"]);
    device_move(data["fields/braid/values"]);
    device_move(data["fields/radial/values"]);
    device_move(data["fields/vel/values/u"]);
    device_move(data["fields/vel/values/v"]);
    device_move(data["fields/vel/values/w"]);

    ASCENT_ANNOTATE_MARK_END("host_copy_to_device");

    output_file = conduit::utils::join_file_path(output_path,
                                                 "tout_render_gpu_source_data_2_");

    // remove old images before rendering
    remove_test_image(output_file);
    scenes["s1/image_prefix"] = output_file;

    extracts["e1/params/path"] = conduit::utils::join_file_path(output_path,
                                                        "tout_ext_gpu_source_data_2");

    sinfo["file_name"] = conduit::utils::join_file_path(output_path,
                                                        "tout_info_gpu_source_data_2.yaml");

    ASCENT_ANNOTATE_MARK_BEGIN("ascent_already_on_device");
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    ASCENT_ANNOTATE_MARK_END("ascent_already_on_device");

    ASCENT_ANNOTATE_MARK_BEGIN("device_cleanup");
    device_cleanup(data["coordsets/coords/values/x"]);
    device_cleanup(data["coordsets/coords/values/y"]);
    device_cleanup(data["coordsets/coords/values/z"]);
    device_cleanup(data["topologies/mesh/elements/connectivity"]);
    device_cleanup(data["fields/braid/values"]);
    device_cleanup(data["fields/radial/values"]);
    device_cleanup(data["fields/vel/values/u"]);
    device_cleanup(data["fields/vel/values/v"]);
    device_cleanup(data["fields/vel/values/w"]);

    ASCENT_ANNOTATE_MARK_END("device_cleanup");

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    ascent::annotations::flush();
    ascent::annotations::finalize();

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


