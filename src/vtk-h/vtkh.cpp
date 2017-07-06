#include "vtkh.hpp"
#include "vtkh_error.hpp"

#include <vtkm/cont/RuntimeDeviceInformation.h>
#include <vtkm/cont/DeviceAdapterListTag.h>
#include <sstream>

namespace vtkh
{

#ifdef PARALLEL

MPI_Comm VTKH::m_mpi_comm = NULL;

void 
VTKH::Open(MPI_Comm mpi_comm)
{
  m_mpi_comm = mpi_comm;
}

MPI_Comm 
VTKH::GetMPIComm()
{
  if(m_mpi_comm == NULL)
  {
    std::stringstream msg;
    msg<<"VTK-h internal error. There is no valid MPI comm availible. ";
    msg<<"It is likely that VTKH.Open(MPI_Comm) was not called.";
    throw Error(msg.str());
  }
  return m_mpi_comm;
}

int 
VTKH::GetMPIRank()
{
  if(m_mpi_comm == NULL)
  {
    std::stringstream msg;
    msg<<"VTK-h internal error. There is no valid MPI comm availible. ";
    msg<<"It is likely that VTKH.Open(MPI_Comm) was not called.";
    throw Error(msg.str());
  }
  int rank;
  MPI_Comm comm = VTKH::GetMPIComm(); 
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int 
VTKH::GetMPISize()
{
  if(m_mpi_comm == NULL)
  {
    std::stringstream msg;
    msg<<"VTK-h internal error. There is no valid MPI comm availible. ";
    msg<<"It is likely that VTKH.Open(MPI_Comm) was not called.";
    throw Error(msg.str());
  }
  int size;
  MPI_Comm comm = VTKH::GetMPIComm(); 
  MPI_Comm_size(comm, &size);
  return size;
}

#else
void VTKH::Open()
{

}
#endif

void VTKH::Close()
{

}

std::string VTKH::About()
{
  std::stringstream msg;
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagCuda> cuda;
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagTBB> tbb;
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagSerial> serial;
  msg<<"---------------- VTK-h -------------------\n";
#ifdef PARALLEL
  int version, subversion;
  MPI_Get_version(&version, &subversion);
  msg<<"MPI version: "<<version<<"."<<subversion<<"\n";
#else
  msg<<"MPI version: n/a\n";
#endif
  msg<<"VTK-m adapters: ";

  if(cuda.Exists())
  {
    msg<<"Cuda ";
  }

  if(tbb.Exists())
  {
    msg<<"TBB ";
  }

  if(serial.Exists())
  {
    msg<<"Serial ";
  }
  msg<<"\n";
 msg<<"------------------------------------------\n";
  return msg.str();
}

}
