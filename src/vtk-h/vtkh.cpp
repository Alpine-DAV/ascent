#include "vtkh.hpp"

#include <vtkm/cont/RuntimeDeviceInformation.h>
#include <vtkm/cont/DeviceAdapterListTag.h>
#include <sstream>

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
