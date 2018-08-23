//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: t_ascent_render_3d.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <runtimes/ascent_vtkh_data_adapter.hpp>
#include <vtkm/cont/testing/MakeTestDataSet.h>
#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_uniform_2d_to_blueprint)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    vtkm::cont::testing::MakeTestDataSet maker; 
    vtkm::cont::DataSet ds = maker.Make2DUniformDataSet0(); 
    conduit::Node blueprint; 
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_uniform_3d_to_blueprint)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    vtkm::cont::testing::MakeTestDataSet maker; 
    vtkm::cont::DataSet ds = maker.Make3DUniformDataSet0(); 
    conduit::Node blueprint; 
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}

//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_rectilinear_3d_to_blueprint)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    vtkm::cont::testing::MakeTestDataSet maker; 
    vtkm::cont::DataSet ds = maker.Make3DRectilinearDataSet0(); 
    conduit::Node blueprint; 
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_rectilinear_2d_to_blueprint)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    vtkm::cont::testing::MakeTestDataSet maker; 
    vtkm::cont::DataSet ds = maker.Make2DRectilinearDataSet0(); 
    conduit::Node blueprint; 
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}

//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_explicit_single_type_to_blueprint)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    vtkm::cont::testing::MakeTestDataSet maker; 
    vtkm::cont::DataSet ds = maker.Make3DExplicitDataSetCowNose(); 
    conduit::Node blueprint;
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, consistent_domain_ids_check)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    
    Node multi_dom; 
    Node &mesh1 = multi_dom.append();
    Node &mesh2 = multi_dom.append();
    conduit::blueprint::mesh::examples::braid("hexs",
                                              2,
                                              2,
                                              2,
                                              mesh1);
    conduit::blueprint::mesh::examples::braid("hexs",
                                              2,
                                              2,
                                              2,
                                              mesh2);
    mesh1.remove("state");
    mesh2["state/domain_id"] = 1;
    bool consistent_ids = false;


    //
    // Publish inconsistent ids to Ascent
    //
    
    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    EXPECT_THROW(ascent.publish(multi_dom),conduit::Error);
    ascent.close();
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


