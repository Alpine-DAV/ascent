//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/Scene.hpp>

#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_clip, vtkh_box_clip)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
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
  vtkm::Vec<vtkm::Float64, 3> center = clip_bounds.Center();
  clip_bounds.X.Max = center[0] + .5;
  clip_bounds.Y.Max = center[1] + .5;
  clip_bounds.Z.Max = center[2] + .5;

  vtkh::Clip clipper;

  clipper.SetBoxClip(clip_bounds);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data_Float64");
  //clipper.AddMapField("cell_data_Float64");
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
                                         "box_clip",
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

TEST(vtkh_clip, vtkh_2_plane_clip)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
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
  vtkm::Vec<vtkm::Float64, 3> center = clip_bounds.Center();
  clip_bounds.X.Max = center[0] + .5;
  clip_bounds.Y.Max = center[1] + .5;
  clip_bounds.Z.Max = center[2] + .5;

  double origin[3] = {center[0], center[1], center[2]};
  double normal1[3] = {0., 0.7, 0.7};
  double normal2[3] = {0., 0., 1.};

  vtkh::Clip clipper;

  clipper.Set2PlaneClip(origin, normal1, origin, normal2);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data_Float64");
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.Azimuth(55.f);
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *clip_output,
                                         "plane2_clip",
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

TEST(vtkh_clip, vtkh_2_plane_clip_particles)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  vtkh::DataSet data_set;

  const int size = 1000;

  data_set.AddDomain(CreateTestDataPoints(size), 0);

  //
  // chop the data set at the center
  //
  vtkm::Bounds clip_bounds = data_set.GetGlobalBounds();
  vtkm::Vec<vtkm::Float64, 3> center = clip_bounds.Center();
  clip_bounds.X.Max = center[0] + .5;
  clip_bounds.Y.Max = center[1] + .5;
  clip_bounds.Z.Max = center[2] + .5;

  double origin[3] = {center[0], center[1], center[2]};
  double normal1[3] = {0., 0.7, 0.7};
  double normal2[3] = {0., 0., 1.};

  vtkh::Clip clipper;

  clipper.Set2PlaneClip(origin, normal1, origin, normal2);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data_Float64");
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.Azimuth(55.f);
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *clip_output,
                                         "plane2_clip_particles",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::PointRenderer renderer;
  renderer.SetInput(clip_output);
  renderer.SetField("point_data_Float64");
  renderer.SetBaseRadius(0.1f);

  scene.AddRenderer(&renderer);
  scene.Render();

  delete clip_output;
}
#if 0
TEST(vtkh_clip, vtkh_sphere_clip)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  //
  // chop the data set at the center
  //

  double center[3] = {0,0,0};

  double radius = base_size * num_blocks * 0.5f;

  vtkh::Clip clipper;

  clipper.SetSphereClip(center, radius);
  clipper.SetInput(&data_set);
  clipper.AddMapField("point_data_Float64");
  clipper.AddMapField("cell_data_Float64");
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkm::Bounds bounds = clip_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(32,32,-80));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *clip_output,
                                         "sphere_clip",
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
#endif
