//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//-----------------------------------------------------------------------------
TEST(vtkh_dataset, vtkh_range)
{
  vtkh::VTKH vtkh;
  vtkh::vtkhDataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2; 

  data_set.AddDomain(CreateTestData(0, num_blocks, base_size), 0);
  data_set.AddDomain(CreateTestData(1, num_blocks, base_size), 1);

  vtkm::Bounds data_bounds = data_set.GetBounds();
  
  const double max_val = base_size * num_blocks;
  const double min_val = 0.; 

  std::cout<<data_bounds<<"\n";

  EXPECT_EQ(data_bounds.X.Min, min_val);
  EXPECT_EQ(data_bounds.Y.Min, min_val);
  EXPECT_EQ(data_bounds.Z.Min, min_val);

  EXPECT_EQ(data_bounds.X.Max, max_val);
  EXPECT_EQ(data_bounds.Y.Max, max_val);
  EXPECT_EQ(data_bounds.Z.Max, max_val);

  vtkm::cont::ArrayHandle<vtkm::Range> vec_range;
  vec_range = data_set.GetRange("vector_data");

  EXPECT_EQ(3, vec_range.GetPortalControl().GetNumberOfValues());
  

  vtkm::cont::ArrayHandle<vtkm::Range> scalar_range;
  scalar_range = data_set.GetRange("point_data");
  EXPECT_EQ(1, scalar_range.GetPortalControl().GetNumberOfValues());

  vtkm::Float64 min_coord = 0.;
  vtkm::Float64 max_coord = vtkm::Float64(base_size * num_blocks);
 
  vtkm::Bounds bounds  = data_set.GetBounds();
  EXPECT_EQ(min_coord, bounds.X.Min);
  EXPECT_EQ(min_coord, bounds.Y.Min);
  EXPECT_EQ(min_coord, bounds.Z.Min);
  EXPECT_EQ(max_coord, bounds.X.Max);
  EXPECT_EQ(max_coord, bounds.Y.Max);
  EXPECT_EQ(max_coord, bounds.Z.Max);
}
