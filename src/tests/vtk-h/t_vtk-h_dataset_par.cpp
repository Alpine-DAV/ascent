//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//-----------------------------------------------------------------------------
TEST(vtkh_dataset_par, vtkh_range_par)
{
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  vtkh::VTKH vtkh;
  vtkh.Open(MPI_COMM_WORLD);
  vtkh::vtkhDataSet data_set;
 
  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank; 
  
  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }

  vtkm::Bounds data_bounds = data_set.GetGlobalBounds();
  
  const double max_val = base_size * num_blocks;
  const double min_val = 0.; 

  EXPECT_EQ(data_bounds.X.Min, min_val);
  EXPECT_EQ(data_bounds.Y.Min, min_val);
  EXPECT_EQ(data_bounds.Z.Min, min_val);

  EXPECT_EQ(data_bounds.X.Max, max_val);
  EXPECT_EQ(data_bounds.Y.Max, max_val);
  EXPECT_EQ(data_bounds.Z.Max, max_val);

  std::cout<<data_bounds<<"\n";

  vtkm::cont::ArrayHandle<vtkm::Range> vec_range;

  vec_range = data_set.GetGlobalRange("vector_data");

  EXPECT_EQ(3, vec_range.GetPortalControl().GetNumberOfValues());

  vtkm::cont::ArrayHandle<vtkm::Range> scalar_range;
  scalar_range = data_set.GetGlobalRange("point_data");
  EXPECT_EQ(1, scalar_range.GetPortalControl().GetNumberOfValues());

  MPI_Finalize();
}
