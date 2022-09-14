//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>

//-----------------------------------------------------------------------------
TEST(vtkh_dataset_par, vtkh_range_par)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));
  vtkh::DataSet data_set;

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

  vec_range = data_set.GetGlobalRange("vector_data_Float64");

  EXPECT_EQ(3, vec_range.GetNumberOfValues());

  vtkm::cont::ArrayHandle<vtkm::Range> scalar_range;
  scalar_range = data_set.GetGlobalRange("point_data_Float64");
  EXPECT_EQ(1, scalar_range.GetNumberOfValues());

  int topo_dims;
  EXPECT_EQ(true, data_set.IsStructured(topo_dims));
  EXPECT_EQ(3, topo_dims);

  if(rank ==0)
  {
    vtkm::cont::DataSet unstructured;

    std::vector<vtkm::Vec<vtkm::Float32,3>> coords;
    coords.push_back(vtkm::Vec<vtkm::Float32,3>(0.f, 0.f, 0.f));
    coords.push_back(vtkm::Vec<vtkm::Float32,3>(1.f, 0.f, 0.f));
    coords.push_back(vtkm::Vec<vtkm::Float32,3>(0.f, 0.f, 1.f));

    std::vector<vtkm::UInt8> shapes;
    shapes.push_back(5);
    std::vector<vtkm::IdComponent> num_indices;
    num_indices.push_back(3);
    std::vector<vtkm::Id> conn;
    conn.push_back(0);
    conn.push_back(1);
    conn.push_back(2);

    vtkm::cont::DataSetBuilderExplicit builder;
    unstructured = builder.Create(coords, shapes, num_indices, conn, "coordinates");
    data_set.AddDomain(unstructured, -1);
  }

  EXPECT_EQ(false, data_set.IsStructured(topo_dims));
  EXPECT_EQ(-1, topo_dims);

  MPI_Finalize();
}
