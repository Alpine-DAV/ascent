//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_point_renderer, vtkh_point_render)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
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
                                         "render_points");
  vtkh::PointRenderer renderer;
  renderer.SetInput(&data_set);
  renderer.SetField("point_data_Float64");



  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

}

TEST(vtkh_point_renderer, vtkh_variable_point_render)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
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
                                         "render_var_points");
  vtkh::PointRenderer renderer;
  renderer.SetInput(&data_set);
  renderer.SetField("point_data_Float64");
  renderer.UseVariableRadius(true);
  renderer.SetRadiusDelta(1.0f);



  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

}

TEST(vtkh_point_renderer, vtkh_point_no_data)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 16;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }
  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 1;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();
  data_set = *iso_output;
  delete iso_output;

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16, 36, -36));
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "render_no_data");
  vtkh::PointRenderer renderer;
  renderer.SetInput(&data_set);
  renderer.SetField("point_data_Float64");
  renderer.UseVariableRadius(true);
  renderer.SetRadiusDelta(1.0f);



  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

}
