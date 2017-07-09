#ifndef VTK_H_HPP
#define VTK_H_HPP

#include <string>

#ifdef PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

class VTKH
{
private:
#ifdef PARALLEL
  static MPI_Comm m_mpi_comm;
#endif
public:
#ifdef PARALLEL
  void Open(MPI_Comm mpi_comm);
  static MPI_Comm GetMPIComm();
  static int GetMPIRank();
  static int GetMPISize();
#else
  void Open();
#endif
  void Close();
  std::string About();
};

}
#endif
