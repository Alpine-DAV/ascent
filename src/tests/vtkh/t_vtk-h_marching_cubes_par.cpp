//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>


//----------------------------------------------------------------------------
TEST(vtkh_marching_cubes_par, vtkh_parallel_marching_cubes)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif

  MPI_Init(NULL, NULL);
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

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");

  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *iso_output = marcher.GetOutput();
  vtkm::Bounds bounds = iso_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);

  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *iso_output,
                                         "iso_par",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("cell_data_Float64");
  scene.AddRenderer(&tracer);

  scene.Render();

  delete iso_output;
  MPI_Finalize();
}
