//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/HistSampling.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>

//----------------------------------------------------------------------------
TEST(vtkh_hist_sampling_par, vtkh_sampling_point_view)
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

  vtkh::HistSampling sampler;

  sampler.SetField("point_data_Float64");
  sampler.SetGhostField("ghosts");
  sampler.SetInput(&data_set);
  sampler.SetSamplingPercent(0.01);

  sampler.Update();
  vtkh::DataSet *output = sampler.GetOutput();

  vtkm::Bounds bounds = output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *output,
                                         "sample_points");
  vtkh::PointRenderer renderer;
  renderer.SetInput(output);
  renderer.SetField("valSampled");
  renderer.UseVariableRadius(true);
  renderer.SetRadiusDelta(2.0f);

  vtkh::Scene scene;
  scene.AddRenderer(&renderer);
  scene.AddRender(render);
  scene.Render();

  delete output;
}

//----------------------------------------------------------------------------
TEST(vtkh_hist_sampling_par, vtkh_sampling_cell_view)
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

  vtkh::HistSampling sampler;

  sampler.SetField("point_data_Float64");
  sampler.SetGhostField("ghosts");
  sampler.SetInput(&data_set);
  sampler.SetSamplingPercent(0.01);

  sampler.Update();
  vtkh::DataSet *output = sampler.GetOutput();

  vtkm::Bounds bounds = output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *output,
                                         "sample_points_cell_view");
  vtkh::RayTracer tracer;

  tracer.SetInput(output);
  tracer.SetField("point_data_Float64");
  //tracer.SetField("valSampled");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete output;

}

int main(int argc, char* argv[])
{
    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
