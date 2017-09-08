#ifndef VTK_H_HPP
#define VTK_H_HPP

#include <string>

#ifdef PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

  std::string AboutVTKH();
  bool IsSerialEnabled();
  bool IsTBBEnabled();
  bool IsCUDAEnabled();
  void ForceSerial();
  void ForceTBB();
  void ForceCUDA();
  void ResetDevices();
#ifdef PARALLEL
  void   SetMPIComm(MPI_Comm mpi_comm);
  MPI_Comm GetMPIComm();
  int GetMPIRank();
  int GetMPISize();
#endif

}
#endif
