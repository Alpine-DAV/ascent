//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_serial_render)
{
  vtkh::vtkhDataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }
  vtkh::vtkhRayTracer tracer;
   
  tracer.SetInput(&data_set);
  tracer.SetField("point_data"); 
  tracer.Update();
 
}
