//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Histogram.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>
#include <mpi.h>

//----------------------------------------------------------------------------
TEST(vtkh_histogram_par, vtkh_histogram_clamp_range)
{

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

  vtkh::Histogram::HistogramResult res;
  vtkh::Histogram histogrammer;
  histogrammer.SetNumBins(128);

  vtkm::Range range;
  range.Min = 0;
  range.Max = 100;

  histogrammer.SetRange(range);
  res = histogrammer.Run(data_set,"point_data_Float64");

  if(rank == 0) res.Print(std::cout);

  MPI_Finalize();
}
