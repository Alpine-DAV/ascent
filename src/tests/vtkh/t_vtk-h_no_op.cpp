//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_no_op, vtkh_serial_no_op)
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

  vtkh::NoOp noop;
  noop.SetInput(&data_set);
  noop.SetField("point_data_Float64");

  noop.Update();

  vtkh::DataSet *noop_output = noop.GetOutput();

  vtkm::Bounds bounds = noop_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *noop_output,
                                         "noop",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(noop_output);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete noop_output;
}
