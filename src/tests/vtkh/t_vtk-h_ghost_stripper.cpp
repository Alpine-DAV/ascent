//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>

#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_ghost_stripper, vtkh_ghost_stripper)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  //
  // chop the data set at the center
  //
  vtkm::Bounds clip_bounds = data_set.GetGlobalBounds();

  vtkh::GhostStripper stripper;

  stripper.SetInput(&data_set);
  stripper.SetField("ghosts");
  stripper.AddMapField("point_data_Float64");
  stripper.Update();

  vtkh::DataSet *stripped_output = stripper.GetOutput();

  vtkm::Bounds bounds = stripped_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *stripped_output,
                                         "ghost_stipper",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(stripped_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete stripped_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_ghost_stripper, vtkh_ghost_stripper_no_strip)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Id before_cells = data_set.GetNumberOfCells();

  vtkh::GhostStripper stripper;

  stripper.SetInput(&data_set);
  stripper.SetField("ghosts");
  stripper.SetMinValue(0);
  stripper.SetMaxValue(2);
  stripper.AddMapField("point_data_Float64");
  stripper.Update();

  vtkh::DataSet *stripped_output = stripper.GetOutput();

  vtkm::Id after_cells = stripped_output->GetNumberOfCells();

  assert(before_cells == after_cells);
  delete stripped_output;
}
