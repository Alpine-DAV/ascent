//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_slice.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>

//-----------------------------------------------------------------------------
TEST(vtkh_slice, vtkh_slice)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::Slice slicer;

  vtkm::Vec<vtkm::Float32,3> normal(.5f,.5f,.5f);
  vtkm::Vec<vtkm::Float32,3> point(16.f,16.f,16.f);
  slicer.AddPlane(point, normal);
  slicer.SetInput(&data_set);
  slicer.Update();
  vtkh::DataSet *slice  = slicer.GetOutput();

  vtkm::Bounds bounds = slice->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *slice,
                                         "slice",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(slice);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete slice;
}

//-----------------------------------------------------------------------------
TEST(vtkh_slice, vtkh_mulit_slice)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkm::Bounds bounds;
  vtkh::Scene scene;

  // add the first slice
  vtkh::Slice slicer1;

  vtkm::Vec<vtkm::Float32,3> normal1(.0f,5.f,.5f);
  vtkm::Vec<vtkm::Float32,3> point1(16.f,16.f,16.f);
  vtkm::Vec<vtkm::Float32,3> normal2(.5f,.5f,.5f);
  vtkm::Vec<vtkm::Float32,3> point2(16.f,16.f,16.f);
  slicer1.AddPlane(point1, normal1);
  slicer1.AddPlane(point2, normal2);
  slicer1.SetInput(&data_set);
  slicer1.Update();
  vtkh::DataSet *slice1  = slicer1.GetOutput();
  bounds.Include(slice1->GetGlobalBounds());

  vtkh::RayTracer tracer1;
  tracer1.SetInput(slice1);
  tracer1.SetField("cell_data_Float64");
  scene.AddRenderer(&tracer1);


  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};

  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         bounds,
                                         "2slice",
                                          bg_color);

  scene.AddRender(render);
  scene.Render();

  delete slice1;
}

//---------------------------------------------------------------------------//
TEST(vtkh_slice, vtkh_slice_implicit_sphere)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::SliceImplicit slicer;
  double center[3] = { 0.0, 0.0, 0.0};
  
  slicer.SetSphereSlice(center, 10.0);
  slicer.SetInput(&data_set);
  slicer.Update();
  vtkh::DataSet *slice  = slicer.GetOutput();

  vtkm::Bounds bounds = slice->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *slice,
                                         "slice_implicit_sphere",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(slice);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete slice;
}

//---------------------------------------------------------------------------//
TEST(vtkh_slice, vtkh_slice_implicit_cylinder)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::SliceImplicit slicer;

  // void SetCylinderSlice(const double center[3],
  //                       const double axis[3],
  //                       const double radius);

  double center[3] = { 0.0, 0.0, 0.0};
  double axis[3]   = { 0.0, 0.0, 1.0};

  slicer.SetCylinderSlice(center,
                          axis,
                          10.0);

  slicer.SetInput(&data_set);
  slicer.Update();
  vtkh::DataSet *slice  = slicer.GetOutput();

  vtkm::Bounds bounds = slice->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *slice,
                                         "slice_implicit_cylinder",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(slice);
  tracer.SetField("cell_data_Float64");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete slice;
}

//---------------------------------------------------------------------------//
TEST(vtkh_slice, vtkh_slice_implicit_box)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::SliceImplicit slicer;

  vtkm::Bounds slice_bounds(vtkm::Range(0, 20),
                            vtkm::Range(0, 15),
                            vtkm::Range(-1, 5));

  // note: we keep one face "open"
  slicer.SetBoxSlice(slice_bounds);
  slicer.SetInput(&data_set);
  slicer.Update();
  vtkh::DataSet *slice  = slicer.GetOutput();

  vtkm::Bounds bounds = slice->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *slice,
                                         "slice_implicit_box",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(slice);
  tracer.SetField("cell_data_Float64");



  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete slice;
}



  // void Set3PlaneSlice(const double origin1[3],
  //                     const double normal1[3],
  //                     const double origin2[3],
  //                     const double normal2[3],
  //                     const double origin3[3],
  //                     const double normal3[3]);

//---------------------------------------------------------------------------//
TEST(vtkh_slice, vtkh_slice_implicit_plane)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::SliceImplicit slicer;

  double origin[3] = {16.0,16.0,16.0};
  double normal[3] = {0.5,0.5,0.5};

  // void SetPlaneSlice(const double origin[3], const double normal[3]);

  slicer.SetPlaneSlice(origin,normal);
  slicer.SetInput(&data_set);
  slicer.Update();
  vtkh::DataSet *slice  = slicer.GetOutput();

  vtkm::Bounds bounds = slice->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *slice,
                                         "slice_implicit_plane",
                                          bg_color);
  vtkh::RayTracer tracer;
  tracer.SetInput(slice);
  tracer.SetField("cell_data_Float64");



  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete slice;
}


//
// TODO: Multi Plane Implicit needs work.
//

// //---------------------------------------------------------------------------//
// TEST(vtkh_slice, vtkh_slice_implicit_2plane)
// {
// #ifdef VTKM_ENABLE_KOKKOS
//   vtkh::InitializeKokkos();
// #endif
//   vtkh::DataSet data_set;
//
//   const int base_size = 32;
//   const int num_blocks = 1;
//
//   for(int i = 0; i < num_blocks; ++i)
//   {
//     data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
//   }
//
//   vtkh::SliceImplicit slicer;
//
//   double origin_a[3] = {16.0,16.0,16.0};
//   double normal_a[3] = {0.5,0.5,0.5};
//
//   double origin_b[3] = {16.0,16.0,16.0};
//   double normal_b[3] = {0.0,0.5,0.5};
//
//   // void Set2PlaneSlice(const double origin1[3],
//   //                     const double normal1[3],
//   //                     const double origin2[3],
//   //                     const double normal2[3]);
//
//   slicer.Set2PlaneSlice(origin_a,normal_a,
//                         origin_b,normal_b);
//   slicer.SetInput(&data_set);
//   slicer.Update();
//   vtkh::DataSet *slice  = slicer.GetOutput();
//
//   vtkm::Bounds bounds = slice->GetGlobalBounds();
//   float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
//   vtkm::rendering::Camera camera;
//   camera.ResetToBounds(bounds);
//   vtkh::Render render = vtkh::MakeRender(512,
//                                          512,
//                                          camera,
//                                          *slice,
//                                          "slice_implicit_2plane",
//                                           bg_color);
//   vtkh::RayTracer tracer;
//   tracer.SetInput(slice);
//   tracer.SetField("cell_data_Float64");
//
//
//
//   vtkh::Scene scene;
//   scene.AddRenderer(&tracer);
//   scene.AddRender(render);
//   scene.Render();
//
//   delete slice;
// }


