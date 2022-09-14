//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/ScalarRenderer.hpp>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>


//----------------------------------------------------------------------------
TEST(vtkh_scalar_renderer, vtkh_parallel_render)
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

  vtkm::Bounds bounds = data_set.GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.Azimuth(-30.f);
  camera.Elevation(-30.f);

  vtkh::ScalarRenderer tracer;

  tracer.SetInput(&data_set);
  tracer.SetCamera(camera);
  tracer.Update();

  vtkh::DataSet *output = tracer.GetOutput();

  if(vtkh::GetMPIRank() == 0)
  {
    vtkm::cont::DataSet &result = output->GetDomain(0);
    vtkm::io::VTKDataSetWriter writer("scalar_data.vtk");
    writer.WriteDataSet(result);
  }

  MPI_Finalize();
}
