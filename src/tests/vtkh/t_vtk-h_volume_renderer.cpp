//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render_ustructured)
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

  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10.;
  iso_range.Max = 40.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  vtkm::Vec<vtkm::Float32,3> pos = camera.GetPosition();
  pos[0]+=.1;
  pos[1]+=.1;
  camera.SetPosition(pos);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "volume_unstructured");


  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  color_map.AddPointAlpha(1.0, 0.6);

  vtkh::VolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}

TEST(vtkh_volume_renderer, vtkh_parallel_render)
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

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  vtkm::Vec<vtkm::Float32,3> pos = camera.GetPosition();
  pos[0]+=.1;
  pos[1]+=.1;
  camera.SetPosition(pos);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "volume");


  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  color_map.AddPointAlpha(1.0, 0.6);

  vtkh::VolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}
