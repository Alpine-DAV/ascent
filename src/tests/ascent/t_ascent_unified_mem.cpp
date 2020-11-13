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
/// file: t_ascent_slice.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include "nvToolsExt.h"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_log, test_log)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
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
    data["fields"].remove_child("vel");
    data["fields"].schema().print();

    nvtxRangePushA("memory_alloc");
    double * braid = data["fields/braid/values"].as_float64_ptr();
    int size_braid = data["fields/braid/values"].dtype().number_of_elements();

    double * radial = data["fields/radial/values"].as_float64_ptr();
    int size_radial = data["fields/radial/values"].dtype().number_of_elements();

    double * unified_braid;
    double * device_braid;
    cudaMallocManaged(&unified_braid, size_braid*sizeof(double));
    cudaMalloc(&device_braid, size_braid*sizeof(double));
    cudaMemcpy(device_braid, braid,sizeof(double) * size_braid, cudaMemcpyHostToDevice);
    cudaMemcpy(unified_braid, device_braid,sizeof(double) * size_braid, cudaMemcpyDeviceToDevice);
    data["fields/braid/values"].set_external(unified_braid, size_braid);

    double * unified_radial;
    double * device_radial;
    cudaMallocManaged(&unified_radial, size_radial*sizeof(double));
    cudaMalloc(&device_radial, size_radial*sizeof(double));
    cudaMemcpy(device_radial, radial,sizeof(double) * size_radial, cudaMemcpyHostToDevice);
    cudaMemcpy(unified_radial, device_radial,sizeof(double) * size_radial, cudaMemcpyDeviceToDevice);
    data["fields/radial/values"].set_external(unified_radial, size_radial);

    nvtxRangePop();


    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    ASCENT_INFO("Testing unified mem");
    data["fields"].schema().print();


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_unified_mem");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    nvtxRangePushA("ascent");
    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    nvtxRangePop();

    // check that we created an image
    //EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the log filter.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

    cudaFree(unified_braid);
    cudaFree(device_braid);
    cudaFree(unified_radial);
    cudaFree(device_radial);
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


