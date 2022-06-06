//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>

#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_iso_volume, vtkh_iso_volume)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10.;
  iso_range.Max = 30.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso_volume",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete iso_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_iso_volume, vtkh_iso_volume_empty)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 40e16;
  iso_range.Max = 30e16;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso_volume_empty",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete iso_output;
}
