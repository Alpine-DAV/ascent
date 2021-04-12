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
/// file: t_ascent_cinema_a.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <runtimes/expressions/ascent_blueprint_architect.hpp>
#include <runtimes/expressions/ascent_execution.hpp>
#include <runtimes/expressions/ascent_memory_manager.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"



using namespace std;
using namespace conduit;
using namespace ascent;

// do not change this
const index_t EXAMPLE_MESH_SIDE_DIM = 32;

// Only convert leaf arrays to GPU memory. Examples
// of things we would not want on the GPU are field association
// strings
void device_conversion(Node &host_data, Node &device_data)
{
  const int32 children = host_data.number_of_children();
  if(children == 0)
  {
    if(host_data.dtype().is_number() && host_data.dtype().number_of_elements() > 1)
    {
      // we only want to set device mem for arrays
      device_data.set_allocator(AllocationManager::conduit_device_allocator_id());
      std::cout<<host_data.to_summary_string()<<"\n";
    }
    device_data = host_data;
    return;
  }
  std::vector<std::string> names = host_data.child_names();
  for(auto &name : names)
  {
    device_conversion(host_data[name], device_data[name]);
  }
}
#if 0
//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, max)
{

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ExecutionManager::execution("cuda");
    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    data["state/domain_id"] = 0;
    Node dataset;
    dataset.append().set_external(data);

    Node res = runtime::expressions::field_max(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  9.98820080464372, 0.0001);
    EXPECT_EQ(res["index"].to_int32(), 817);
}

//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, sum)
{

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    data["state/domain_id"] = 0;
    Node dataset;
    dataset.append().set_external(data);

    Node res = runtime::expressions::field_sum(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  -1082.59582227314, 0.0001);
}

//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, min)
{

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    data["state/domain_id"] = 0;
    Node dataset;
    dataset.append().set_external(data);

    Node res = runtime::expressions::field_min(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  -9.7849527094773894, 0.0001);
    EXPECT_EQ(res["index"].to_int32(), 10393);
}

//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, max_already_gpu)
{
    // this is normally set in ascent::Initialize, but we
    // have to set it here so that we do the right thing with
    // device pointers
    AllocationManager::set_conduit_mem_handlers();

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    data["state/domain_id"] = 0;
    Node device_data;
    device_conversion(data, device_data);

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    // work aournd data we need
    Node dataset;
    dataset.append().set_external(device_data);

    Node res = runtime::expressions::field_max(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  9.98820080464372, 0.0001);
    EXPECT_EQ(res["index"].to_int32(), 817);
}

//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, max_already_gpu_rectilinear)
{
    // this is normally set in ascent::Initialize, but we
    // have to set it here so that we do the right thing with
    // device pointers
    AllocationManager::set_conduit_mem_handlers();

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("rectilinear",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    data["state/domain_id"] = 0;
    Node device_data;
    device_conversion(data, device_data);

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    // work aournd data we need
    Node dataset;
    dataset.append().set_external(device_data);

    Node res = runtime::expressions::field_max(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  9.98820080464372, 0.0001);
    EXPECT_EQ(res["index"].to_int32(), 817);
}

TEST(ascent_blueprint_reductions, max_already_gpu_zone_centered)
{
    // this is normally set in ascent::Initialize, but we
    // have to set it here so that we do the right thing with
    // device pointers
    AllocationManager::set_conduit_mem_handlers();

    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    data["state/domain_id"] = 0;
    Node device_data;
    device_conversion(data, device_data);

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    // work aournd data we need
    Node dataset;
    dataset.append().set_external(device_data);

    Node res = runtime::expressions::field_max(dataset,"radial");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  173.205080756888, 0.0001);
    // Its not obvious to me that the zone would be 0, so if this fails,
    // go check things
    EXPECT_EQ(res["index"].to_int32(), 0);
}


//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, ave)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    data["state/domain_id"] = 0;
    Node dataset;
    dataset.append().set_external(data);

    Node res = runtime::expressions::field_avg(dataset,"braid");
    res.print();
    EXPECT_NEAR(res["value"].to_float64(),  -0.0330382025840188, 0.001);
}
//-----------------------------------------------------------------------------

#endif
TEST(ascent_blueprint_reductions, max_already_gpu_histogram)
{
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    data["state/domain_id"] = 0;
    Node device_data;
    device_conversion(data, device_data);

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    // work aournd data we need
    Node dataset;
    dataset.append().set_external(device_data);

    const int num_bins = 64;
    Node res = runtime::expressions::field_histogram(dataset,"braid", -10, 10, num_bins);
    res.print();
    EXPECT_NEAR(res["value"].as_float64_ptr()[0], 8.0, 0.0001);
}

//-----------------------------------------------------------------------------
TEST(ascent_blueprint_reductions, histogram)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ExecutionManager::execution("cuda");

    // everything expects a mutli-domain data set and expects that there
    // are domain ids
    data["state/domain_id"] = 0;
    Node dataset;
    dataset.append().set_external(data);

    const double min_val = -10.f;
    const double max_val = 10.f;
    constexpr int num_bins = 16;
    Node res = runtime::expressions::field_histogram(dataset,"braid",min_val, max_val, num_bins);
    res.print();
    // right now everything is stored using doubles
    double counts[num_bins] = { 126.0,  524.0, 1035.0, 1582.0,
                               2207.0, 2999.0, 3548.0, 4378.0,
                               4361.0, 3583.0, 2983.0, 2459.0,
                               1646.0,  858.0,  379.0,  100.0};

    double *vals = res["value"].as_float64_ptr();
    for(int i = 0; i < num_bins; ++i)
    {
      EXPECT_EQ(counts[i], vals[i]);
    }

    //EXPECT_NEAR(res["value"].to_float64(),  -0.0330382025840188, 0.001);
}

int
main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // this is normally set in ascent::Initialize, but we
  // have to set it here so that we do the right thing with
  // device pointers
  AllocationManager::set_conduit_mem_handlers();

  result = RUN_ALL_TESTS();
  return result;
}
