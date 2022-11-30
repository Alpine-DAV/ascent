//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_particle_advection_par.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Streamline.hpp>
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/cont/CellSetSingleType.h>
#include "t_vtkm_test_utils.hpp"
#include <iostream>
#include <mpi.h>

void checkValidity(vtkh::DataSet *data, const int maxSteps, bool isSL)
{
  int numDomains = data->GetNumberOfDomains();

  //Check all domains
  for(int i = 0; i < numDomains; i++)
  {
    auto currentDomain = data->GetDomain(i);
    auto cs = currentDomain.GetCellSet();
    if (isSL)
    {
      auto cellSet = cs.AsCellSet<vtkm::cont::CellSetExplicit<>>();
      //Ensure that streamlines took <= to the max number of steps
      for(int j = 0; j < cellSet.GetNumberOfCells(); j++)
      {
        EXPECT_LE(cellSet.GetNumberOfPointsInCell(j), maxSteps);
      }
    }
    else
    {
      if (!cs.IsType<vtkm::cont::CellSetSingleType<>>())
        EXPECT_TRUE(false);
    }
  }
}

void writeDataSet(vtkh::DataSet *data, std::string fName, int rank)
{
  int numDomains = data->GetNumberOfDomains();
  std::cerr << "num domains " << numDomains << std::endl;
  for(int i = 0; i < numDomains; i++)
  {
    char fileNm[128];
    sprintf(fileNm, "%s.rank%d.domain%d.vtk", fName.c_str(), rank, i);
    vtkm::io::VTKDataSetWriter write(fileNm);
    write.WriteDataSet(data->GetDomain(i));
  }
}

static inline vtkm::FloatDefault
rand01()
{
  return (vtkm::FloatDefault)rand() / (RAND_MAX+1.0f);
}

static inline vtkm::FloatDefault
randRange(const vtkm::FloatDefault &a, const vtkm::FloatDefault &b)
{
    return a + (b-a)*rand01();
}

template <typename FilterType>
vtkh::DataSet *
RunFilter(vtkh::DataSet& input,
          const std::string& fieldName,
          const std::vector<vtkm::Particle>& seeds,
          int maxAdvSteps,
          double stepSize)
{
  FilterType filter;

  filter.SetInput(&input);
  filter.SetField(fieldName);
  filter.SetNumberOfSteps(maxAdvSteps);
  filter.SetStepSize(stepSize);
  filter.SetSeeds(seeds);
  filter.Update();

  return filter.GetOutput();
}

//----------------------------------------------------------------------------
TEST(vtkh_particle_advection, vtkh_serial_particle_advection)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  std::cout << "Running parallel Particle Advection, vtkh - with " << comm_size << " ranks" << std::endl;

  vtkh::DataSet data_set;
  const int base_size = 32;
  const int blocks_per_rank = 1;
  const int maxAdvSteps = 1000;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestDataRectilinear(domain_id, num_blocks, base_size), domain_id);
  }

  std::vector<vtkm::Particle> seeds;

  vtkm::Bounds bounds = data_set.GetGlobalBounds();
  std::cout<<"Bounds= "<<bounds<<std::endl;

  for (int i = 0; i < 100; i++)
  {
    vtkm::Particle p;
    p.Pos = vtkm::Vec3f(randRange(bounds.X.Min, bounds.X.Max),
                        randRange(bounds.Y.Min, bounds.Y.Max),
                        randRange(bounds.Z.Min, bounds.Z.Max));
    p.ID = static_cast<vtkm::Id>(i);

    seeds.push_back(p);
  }

  vtkh::DataSet *outPA=NULL, *outSL=NULL;

  outPA = RunFilter<vtkh::ParticleAdvection>(data_set, "vector_data_Float64", seeds, maxAdvSteps, 0.1);
  outPA->PrintSummary(std::cout);
  checkValidity(outPA, maxAdvSteps+1, false);

  outSL = RunFilter<vtkh::Streamline>(data_set, "vector_data_Float64", seeds, maxAdvSteps, 0.1);
  outSL->PrintSummary(std::cout);
  checkValidity(outSL, maxAdvSteps+1, true);

  writeDataSet(outSL, "advection_SeedsRandomWhole", rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
