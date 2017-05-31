#include "vtkh.hpp"

namespace vtkh
{

#ifdef PARALLEL

MPI_Comm VTKH::m_mpi_comm;

void VTKH::Open(MPI_Comm mpi_comm)
{
  m_mpi_comm = mpi_comm;
}

MPI_Comm VTKH::GetMPIComm()
{
  return m_mpi_comm;
}

#else
void VTKH::Open()
{

}
#endif

void VTKH::Close()
{

}

}
