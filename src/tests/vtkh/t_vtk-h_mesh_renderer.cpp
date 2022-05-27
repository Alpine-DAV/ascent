//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_mesh_renderer, vtkh_serial_render)
{
  vtkh::DataSet data_set;

  const int base_size = 16;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16, 36, -36));
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "mesh_render_field");
  vtkh::MeshRenderer renderer;
  renderer.SetInput(&data_set);
  renderer.SetField("point_data_Float64");
  renderer.SetUseForegroundColor(true);

  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

}

//----------------------------------------------------------------------------
TEST(vtkh_mesh_renderer, vtkh_serial_render_no_field)
{
  vtkh::DataSet data_set;

  const int base_size = 16;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  data_set.AddConstantPointField(0.f, "constant");
  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16, 36, -36));
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "mesh_render_no_field");
  vtkh::MeshRenderer renderer;
  renderer.SetInput(&data_set);
  renderer.SetField("constant");
  renderer.SetUseForegroundColor(true);

  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

}
