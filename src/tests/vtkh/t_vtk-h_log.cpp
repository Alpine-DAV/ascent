//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_viskores_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_log, vtkh_log)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::Log logger;
  logger.SetInput(&data_set);
  logger.SetField("point_data_Float64");
  logger.SetResultField("log(point_data_Float64)");

  logger.Update();
  vtkh::DataSet *output = logger.GetOutput();
  viskores::Bounds bounds = output->GetGlobalBounds();

  viskores::rendering::Camera camera;
  camera.SetPosition(viskores::Vec<viskores::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *output,
                                         "log");
  vtkh::RayTracer tracer;
  tracer.SetInput(output);
  tracer.SetField("log(point_data_Float64)");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();

  delete output;
}
