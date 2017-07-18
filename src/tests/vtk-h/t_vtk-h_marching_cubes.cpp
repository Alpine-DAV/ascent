//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <vtkh_marching_cubes.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_marching_cubes, vtkh_serial_marching_cubes)
{
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data"); 

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data");
  marcher.AddMapField("cell_data");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data"); 
  tracer.Update();

  delete iso_output; 
}
