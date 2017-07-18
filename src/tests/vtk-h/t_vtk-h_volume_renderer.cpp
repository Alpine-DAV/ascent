//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <rendering/vtkh_renderer_volume.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render)
{
  
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2;
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }
  
  vtkh::vtkhVolumeRenderer tracer;
  vtkm::rendering::ColorTable color_map("cool2warm"); 
  color_map.AddAlphaControlPoint(0.0, .05);
  color_map.AddAlphaControlPoint(1.0, .05);
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data"); 

  tracer.Update();
}
