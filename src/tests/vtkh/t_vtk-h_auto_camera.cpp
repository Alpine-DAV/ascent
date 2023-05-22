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
#include <vtkh/rendering/AutoCamera.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/io/VTKDataSetWriter.h>

#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_auto_camera, vtkh_data_entropy)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int SIZE = 32;
  const int NUM_BLOCKS = 1;
  const int NUM_SAMPLES = 3;

  for(int i = 0; i < NUM_BLOCKS; ++i)
  {
    data_set.AddDomain(CreateTestData(i, NUM_BLOCKS, SIZE), i);
  }

  //
  // chop the data set at the center
  //
  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10.;
  iso_range.Max = 30.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkh::AutoCamera a_camera;

  a_camera.SetMetric("data_entropy");
  a_camera.SetNumSamples(NUM_SAMPLES);
  a_camera.SetInput(iso_output);
  a_camera.SetField("point_data_Float64");
  //a_camera.AddMapField("cell_data_Float64");
  a_camera.Update();

  vtkm::rendering::Camera camera;
  camera = a_camera.GetCamera();

  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};

  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "data_entropy",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

}

//----------------------------------------------------------------------------
TEST(vtkh_auto_camera, vtkh_depth_entropy)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int SIZE = 32;
  const int NUM_BLOCKS = 1;
  const int NUM_SAMPLES = 3;

  for(int i = 0; i < NUM_BLOCKS; ++i)
  {
    data_set.AddDomain(CreateTestData(i, NUM_BLOCKS, SIZE), i);
  }

  //
  // chop the data set at the center
  //
  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10.;
  iso_range.Max = 30.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkh::AutoCamera a_camera;

  a_camera.SetMetric("data_entropy");
  a_camera.SetNumSamples(NUM_SAMPLES);
  a_camera.SetInput(iso_output);
  a_camera.SetField("point_data_Float64");
  //a_camera.AddMapField("cell_data_Float64");
  a_camera.Update();

  vtkm::rendering::Camera camera;
  camera = a_camera.GetCamera();

  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};

  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "depth_entropy",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

}

//----------------------------------------------------------------------------
TEST(vtkh_auto_camera, vtkh_shading_entropy)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int SIZE = 32;
  const int NUM_BLOCKS = 1;
  const int NUM_SAMPLES = 3;

  for(int i = 0; i < NUM_BLOCKS; ++i)
  {
    data_set.AddDomain(CreateTestData(i, NUM_BLOCKS, SIZE), i);
  }

  //
  // chop the data set at the center
  //
  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10.;
  iso_range.Max = 30.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkh::AutoCamera a_camera;

  a_camera.SetMetric("data_entropy");
  a_camera.SetNumSamples(NUM_SAMPLES);
  a_camera.SetInput(iso_output);
  a_camera.SetField("point_data_Float64");
  //a_camera.AddMapField("cell_data_Float64");
  a_camera.Update();

  vtkm::rendering::Camera camera;
  camera = a_camera.GetCamera();

  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};

  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "shading_entropy",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

}

