//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_marching_cubes, vtkh_serial_marching_cubes)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete iso_output;
}
//----------------------------------------------------------------------------
TEST(vtkh_marching_cubes, vtkh_empty)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = -2;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();
  delete iso_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_marching_cubes, vtkh_marching_cubes_recenter)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("cell_data_Float64");

  const int num_vals = 1;
  double iso_vals [num_vals];
  iso_vals[0] = 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso_cell",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete iso_output;
}
