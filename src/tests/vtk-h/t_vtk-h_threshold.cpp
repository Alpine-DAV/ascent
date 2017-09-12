//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <vtkh_threshold.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_threshold, vtkh_serial_threshold)
{
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  data_set.PrintSummary(std::cout);
  vtkh::Threshold thresher;
  thresher.SetInput(&data_set);
  thresher.SetField("point_data"); 

  double upper_bound = (float)base_size * (float)num_blocks * 0.5f;
  double lower_bound = 0;

  thresher.SetUpperThreshold(upper_bound);
  thresher.SetLowerThreshold(lower_bound);
  thresher.AddMapField("point_data");
  thresher.AddMapField("cell_data");
  thresher.Update();
  vtkh::DataSet *output = thresher.GetOutput();
#if 0 
  vtkm::Bounds bounds = output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(512, 
                                                          512, 
                                                          camera, 
                                                          *output, 
                                                          "threshold");  
  // this is outputing a cell set type the 
  // rendering infrastructure has not seen b
  // before
  output->PrintSummary(std::cout);
  vtkh::RayTracer tracer;
  tracer.SetInput(output);
  tracer.AddRender(render);
  tracer.SetField("cell_data"); 
  tracer.Update();
#endif 
  delete output; 
}
