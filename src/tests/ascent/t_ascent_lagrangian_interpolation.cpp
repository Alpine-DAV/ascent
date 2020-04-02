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
/// file: t_ascent_lagrangian.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h> 
#include <cstring>
#include <sstream>
#include <fstream>
#include <string.h>
#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

vtkm::cont::DataSet MakeTestUniformDataSet(vtkm::Id time)
{
  vtkm::Float64 xmin, xmax, ymin, ymax, zmin, zmax;
  xmin = 0.0;
  ymin = 0.0;
  zmin = 0.0;

  xmax = 10.0;
  ymax = 10.0;
  zmax = 10.0;

  const vtkm::Id3 DIMS(16, 16, 16);

  vtkm::cont::DataSetBuilderUniform dsb;

  vtkm::Float64 xdiff = (xmax - xmin) / (static_cast<vtkm::Float64>(DIMS[0] - 1));
  vtkm::Float64 ydiff = (ymax - ymin) / (static_cast<vtkm::Float64>(DIMS[1] - 1));
  vtkm::Float64 zdiff = (zmax - zmin) / (static_cast<vtkm::Float64>(DIMS[2] - 1));

  vtkm::Vec<vtkm::Float64, 3> ORIGIN(0, 0, 0);
  vtkm::Vec<vtkm::Float64, 3> SPACING(xdiff, ydiff, zdiff);

  vtkm::cont::DataSet dataset = dsb.Create(DIMS, ORIGIN, SPACING);
  vtkm::cont::DataSetFieldAdd dsf;

  vtkm::Id numPoints = DIMS[0] * DIMS[1] * DIMS[2];

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>> velocityField;
  velocityField.Allocate(numPoints);

  vtkm::Id count = 0;
  for (vtkm::Id i = 0; i < DIMS[0]; i++)
  {
    for (vtkm::Id j = 0; j < DIMS[1]; j++)
    {
      for (vtkm::Id k = 0; k < DIMS[2]; k++)
      {
        velocityField.WritePortal().Set(count, vtkm::Vec<vtkm::Float64, 3>(0.01, 0.0, 0.0));
        count++;
      }
    }
  }
  dsf.AddPointField(dataset, "velocity", velocityField);
  return dataset;
}

//-----------------------------------------------------------------------------
TEST(ascent_lagrangian_interpolation, test_lagrangian_interpolation_multistep)
{
    //  
    // Set Up MPI
    //  
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);
    vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    ASCENT_INFO("Testing lagrangian interpolation");


//    string output_path = prepare_output_dir();
//    string output_file = conduit::utils::join_file_path(output_path,"tout_lagrangian_3d");

    string output_path = ASCENT_T_BIN_DIR;

    ASCENT_INFO("Execute test from folder: " + output_path + "/ascent");
    output_path = conduit::utils::join_file_path(output_path,"ascent/output");
    ASCENT_INFO("Creating output folder: " + output_path);
    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    // remove old images before rendering
    string output_file1 = conduit::utils::join_file_path(output_path,"Lagrangian_Structured_0_5.vtk");
    string output_file2 = conduit::utils::join_file_path(output_path,"Lagrangian_Structured_0_10.vtk");
    string output_file3 = conduit::utils::join_file_path(output_path,"Pathlines_0_5.vtk");
    string output_file4 = conduit::utils::join_file_path(output_path,"Pathlines_0_10.vtk");
    string output_file5 = conduit::utils::join_file_path(ASCENT_T_BIN_DIR,"/ascent/Seed.txt");
    remove_test_file(output_file1);
    remove_test_file(output_file2);
    remove_test_file(output_file3);
    remove_test_file(output_file4);
    remove_test_file(output_file5);

/*
  Setup - Generate an input dataset. Using VTK-h filter to generate a test dataset. 
*/
    vtkh::Lagrangian lagrangian;
    lagrangian.SetField("velocity");
    lagrangian.SetStepSize(0.1);
    lagrangian.SetWriteFrequency(5);
    lagrangian.SetCustomSeedResolution(1);
    lagrangian.SetSeedResolutionInX(1);
    lagrangian.SetSeedResolutionInY(1);
    lagrangian.SetSeedResolutionInZ(1);
    std::cout << "Running Lagrangian filter to generate data" << std::endl;

  for(vtkm::Id time = 1; time <= 10; ++time)
  {
    vtkh::DataSet data_set;
    data_set.AddDomain(MakeTestUniformDataSet(time),0);
    lagrangian.SetInput(&data_set);
    lagrangian.Update();
    vtkh::DataSet *extracted_basis = lagrangian.GetOutput();
    if(time % 5 == 0)  
    {   
      vtkm::cont::DataSet ds = extracted_basis->GetDomain(0);
    
      std::stringstream file_path;
      file_path << "output/Lagrangian_Structured_0_" << time << ".vtk";
      vtkm::io::writer::VTKDataSetWriter writer(file_path.str());
      writer.WriteDataSet(ds); 
    }   
  }
      
    std::ofstream seedfile;
    seedfile.open("Seed.txt"); 
    seedfile << "5 5 5\n";
    seedfile.close();
  
    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "lagrangian_interpolation";
    // filter knobs
    conduit::Node &lagrangian_params = pipelines["pl1/f1/params"];
    lagrangian_params["field"] = "braid";
    lagrangian_params["radius"] = 0.1;
    lagrangian_params["num_seeds"] = 1;
    lagrangian_params["interval"] = 5; 
    lagrangian_params["start_cycle"] = 5; 
    lagrangian_params["end_cycle"] = 10;
    lagrangian_params["seed_path"] = "Seed.txt";
    lagrangian_params["basis_path"] = "output/";
    lagrangian_params["output_path"] = "output/Pathlines_"; 

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);

    // create dummy mesh using conduit blueprint
    Node n_mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                                2,  
                                                2,  
                                                2,  
                                                n_mesh);
      // publish mesh to ascent                 
    ascent.publish(n_mesh);
    

    actions.append()["action"] = "execute";
    Node &reset  = actions.append();
    reset["action"] = "reset";
    // execute
    ascent.execute(actions);

    ascent.close();

    // check that we created the right output
    EXPECT_TRUE(check_test_file(output_file3));
    EXPECT_TRUE(check_test_file(output_file4));
    std::string msg = "An example of using the lagrangian post hoc interpolation flow filter.";

    // clean up
    remove_test_file(output_file1);
    remove_test_file(output_file2);
    remove_test_file(output_file3);
    remove_test_file(output_file4);
    remove_test_file(output_file5);
    conduit::utils::remove_directory(output_path);

}
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}


