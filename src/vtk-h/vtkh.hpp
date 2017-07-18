#ifndef VTK_H_HPP
#define VTK_H_HPP

#include <string>

#ifdef PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

  std::string AboutVTKH();
#ifdef PARALLEL
  void   SetMPIComm(MPI_Comm mpi_comm);
  MPI_Comm GetMPIComm();
  int GetMPIRank();
  int GetMPISize();
#endif

}
#endif
