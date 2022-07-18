//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>

#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_clip_field, vtkh_clip)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::ClipField clipper;

  clipper.SetClipValue(10.);
  clipper.SetField("point_data_Float64");
  clipper.SetInput(&data_set);
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *clip_output,
                                         "clip_field",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(clip_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete clip_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_clip_field, vtkh_clip_cell_centered)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::ClipField clipper;

  clipper.SetClipValue(.5);
  clipper.SetField("cell_data_Float64");
  clipper.SetInput(&data_set);
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *clip_output,
                                         "clip_field_cell",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(clip_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete clip_output;
}
