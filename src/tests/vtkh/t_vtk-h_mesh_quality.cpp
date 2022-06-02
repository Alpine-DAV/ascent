//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MeshQuality.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/cont/testing/MakeTestDataSet.h>


#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_mesh_quality, vtkh_volume)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  vtkm::cont::testing::MakeTestDataSet maker;
  data_set.AddDomain(maker.Make3DExplicitDataSet5(),0);

  vtkh::MeshQuality quali;
  quali.SetInput(&data_set);
  quali.AddMapField("point_data_Float64");
  quali.AddMapField("cell_data_Float64");
  quali.Update();

  vtkh::DataSet *q_output = quali.GetOutput();

  vtkm::Bounds bounds = q_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *q_output,
                                         "mesh_volume",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(q_output);
  tracer.SetField("volume");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete q_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_mesh_quality, vtkh_not_supported)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  vtkm::cont::testing::MakeTestDataSet maker;
  data_set.AddDomain(maker.Make3DExplicitDataSet5(),0);
  data_set.AddDomain(maker.Make3DUniformDataSet0(),1);

  vtkh::MeshQuality quali;
  quali.SetInput(&data_set);
  quali.AddMapField("point_data_Float64");
  quali.AddMapField("cell_data_Float64");
  bool threw = false;
  try
  {
    quali.Update();
  }
  catch(...)
  {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
