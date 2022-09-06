//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_serial_render)
{
  vtkh::ForceSerial();
  std::cout<<vtkh::AboutVTKH()<<"\n";;

  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "serial_render");
  vtkh::RayTracer tracer;

  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}

//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_omp_render)
{

  if(!vtkh::IsOpenMPAvailable())

  {
    std::cout<<"OpenMP not availible: skipping test.\n";
    return;
  }
  vtkh::ForceOpenMP();
  std::cout<<vtkh::AboutVTKH()<<"\n";;
  ASSERT_TRUE(vtkh::IsOpenMPEnabled());
  ASSERT_FALSE(vtkh::IsSerialEnabled());
  ASSERT_FALSE(vtkh::IsCUDAEnabled());

  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "openmp_render");
  vtkh::RayTracer tracer;

  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}

//----------------------------------------------------------------------------
TEST(vtkh_raytracer, vtkh_cuda_render)
{

  if(!vtkh::IsCUDAAvailable())

  {
    std::cout<<"CUDA not availible: skipping test.\n";
    return;
  }
  vtkh::ForceCUDA();
  std::cout<<vtkh::AboutVTKH()<<"\n";;
  ASSERT_TRUE(vtkh::IsCUDAEnabled());
  ASSERT_FALSE(vtkh::IsOpenMPEnabled());
  ASSERT_FALSE(vtkh::IsSerialEnabled());

  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "openmp_render");
  vtkh::RayTracer tracer;

  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}
