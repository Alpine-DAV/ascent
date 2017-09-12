//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <vtkh_clip.hpp>
#include <rendering/vtkh_renderer_ray_tracer.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
#if 0
TEST(vtkh_clip, vtkh_box_clip)
{
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 1; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }
  data_set.PrintSummary(std::cout);
  //
  // chop the data set at the center
  //
  vtkm::Bounds clip_bounds = data_set.GetGlobalBounds();
  vtkm::Vec<vtkm::Float64, 3> center = clip_bounds.Center();
  clip_bounds.X.Max = center[0] + .5;
  clip_bounds.Y.Max = center[1] + .5;
  clip_bounds.Z.Max = center[2] + .5;

  vtkh::Clip clipper;
  
  clipper.SetBoxClip(clip_bounds);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data");
  clipper.AddMapField("cell_data");
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();
  
  vtkm::Bounds result_bounds = clip_output->GetGlobalBounds();
  std::cout<<"clip_bounds "<<clip_bounds<<" res bounds "<<result_bounds<<"\n";
  clip_output->PrintSummary(std::cout);
  vtkm::cont::DataSet ds = clip_output->GetDomain(0);
  vtkm::cont::CoordinateSystem coord = ds.GetCoordinateSystem();
   
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>> raw;
  //if(coord.GetData().IsSameType(raw))
  //{
  //  std::cout<<"!!!!!\n";
  //  coord.GetData().CopyTo(raw);
  //  auto portal = raw.GetPortalControl();
  //  for(int i = 0; i <raw.GetNumberOfValues(); ++i)
  //  {
  //    std::cout<<portal.Get(i)<<"\n";
  //  }
  //}

  //EXPECT_EQ(, vec_range.GetPortalControl().GetNumberOfValues());
  
  vtkh::vtkhRayTracer tracer;
  tracer.SetInput(clip_output);
  tracer.SetField("point_data"); 
  tracer.Update();
  

  delete clip_output; 
}
#else 
TEST(vtkh_clip, vtkh_sphere_clip)
{
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 2; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }
  data_set.PrintSummary(std::cout);
  //
  // chop the data set at the center
  //
  vtkm::Bounds clip_bounds = data_set.GetGlobalBounds();
  vtkm::Vec<vtkm::Float64, 3> vec_center = clip_bounds.Center();
    
  //double center[3] = {vec_center[0], vec_center[1], vec_center[2]};
  double center[3] = {0,0,0};

  double radius = base_size * num_blocks * 0.5f;

  vtkh::Clip clipper;
  
  clipper.SetSphereClip(center, radius);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data");
  clipper.AddMapField("cell_data");
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();
  
  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(512, 
                                                          512, 
                                                          camera, 
                                                          *clip_output, 
                                                          "clip");  
  vtkh::RayTracer tracer;
  tracer.AddRender(render);
  tracer.SetInput(clip_output);
  tracer.SetField("point_data"); 
  tracer.Update();
  

  delete clip_output; 
}
#endif
