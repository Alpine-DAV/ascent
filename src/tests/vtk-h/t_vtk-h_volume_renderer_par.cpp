//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <rendering/vtkh_renderer_volume.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render)
{
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  vtkh::SetMPIComm(MPI_COMM_WORLD);
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank; 
  
  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }
  
  vtkh::vtkhVolumeRenderer tracer;
  vtkm::rendering::ColorTable color_map("cool2warm"); 
  color_map.AddAlphaControlPoint(0.0, .05);
  color_map.AddAlphaControlPoint(1.0, .05);
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data"); 

  tracer.Update();
 
  MPI_Finalize();
}
