//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>


//----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         data_set,
                                         "volume_par");

  vtkm::cont::ColorTable color_map("cool to warm");
  color_map.AddPointAlpha(0.0, .05);
  color_map.AddPointAlpha(1.0, .5);

  vtkh::VolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();

}

//----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render_unstructured_blank)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
  }

  vtkh::IsoVolume iso;

  vtkm::Range iso_range;
  iso_range.Min = 10000.;
  iso_range.Max = 40000.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();

  vtkm::Bounds bounds = iso_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  vtkm::Vec<vtkm::Float32,3> pos = camera.GetPosition();
  pos[0]+=10000.1;
  pos[1]+=10000.1;
  camera.SetPosition(pos);
  vtkm::Vec<vtkm::Float32,3> look;
  look[0] = 100000.f;
  look[1] = 100000.f;
  look[2] = 100000.f;
  camera.SetLookAt(look);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "volume_unstructured_blank_par");


  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  //color_map.AddPointAlpha(1.0, 0.2);
  color_map.AddPointAlpha(1.0, 0.6);

  vtkh::DataSet blank;
  vtkh::VolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
}
//-----------------------------------------------------------------------------
TEST(vtkh_volume_renderer, vtkh_parallel_render_unstructured)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int blocks_per_rank = 2;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestData(domain_id, num_blocks, base_size), domain_id);
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
                                         "volume_unstructured_par");


  vtkm::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  //color_map.AddPointAlpha(1.0, 0.2);
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
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}


