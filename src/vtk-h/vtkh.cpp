#include "vtkh.hpp"
#include "vtkh_error.hpp"

#include <vtkm/cont/RuntimeDeviceInformation.h>
#include <vtkm/cont/RuntimeDeviceTracker.h>
#include <vtkm/cont/DeviceAdapterListTag.h>
#include <sstream>

namespace vtkh
{
#ifdef PARALLEL

static MPI_Comm g_mpi_comm = MPI_COMM_NULL;

void CheckCommHandle()
{
  if(g_mpi_comm == MPI_COMM_NULL)
  {
    std::stringstream msg;
    msg<<"VTK-h internal error. There is no valid MPI comm availible. ";
    msg<<"It is likely that VTKH.Open(MPI_Comm) was not called.";
    throw Error(msg.str());
  }
}

void 
SetMPIComm(MPI_Comm mpi_comm)
{
  g_mpi_comm = mpi_comm;
}

MPI_Comm 
GetMPIComm()
{
  CheckCommHandle();
  return g_mpi_comm;
}

int 
GetMPIRank()
{
  CheckCommHandle();
  int rank;
  MPI_Comm comm = GetMPIComm(); 
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int 
GetMPISize()
{
  CheckCommHandle();
  int size;
  MPI_Comm comm = GetMPIComm(); 
  MPI_Comm_size(comm, &size);
  return size;
}

#endif

bool IsSerialEnabled()
{
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagSerial> serial;
  return serial.Exists();
}

bool IsTBBEnabled()
{
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagTBB> tbb;
  return tbb.Exists();
}

bool IsCUDAEnabled()
{
  vtkm::cont::RuntimeDeviceInformation<vtkm::cont::DeviceAdapterTagCuda> cuda;
  return cuda.Exists();
}

void ForceSerial()
{
  vtkm::cont::RuntimeDeviceTracker global_tracker;
  global_tracker = vtkm::cont::GetGlobalRuntimeDeviceTracker();
  global_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagSerial());
}

void ForceTBB()
{
  vtkm::cont::RuntimeDeviceTracker global_tracker;
  global_tracker = vtkm::cont::GetGlobalRuntimeDeviceTracker();
  global_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagTBB());
}

void ForceCUDA()
{
  vtkm::cont::RuntimeDeviceTracker global_tracker;
  global_tracker = vtkm::cont::GetGlobalRuntimeDeviceTracker();
  global_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagCuda());
}

void ResetDevices()
{
  vtkm::cont::RuntimeDeviceTracker global_tracker;
  global_tracker = vtkm::cont::GetGlobalRuntimeDeviceTracker();
  global_tracker.Reset();
}

std::string AboutVTKH()
{
  std::stringstream msg;
  msg<<"---------------- VTK-h -------------------\n";
#ifdef PARALLEL
  int version, subversion;
  MPI_Get_version(&version, &subversion);
  msg<<"MPI version: "<<version<<"."<<subversion<<"\n";
#else
  msg<<"MPI version: n/a\n";
#endif
  msg<<"VTK-m adapters: ";

  if(IsCUDAEnabled())
  {
    msg<<"Cuda ";
  }

  if(IsTBBEnabled())
  {
    msg<<"TBB ";
  }

  if(IsSerialEnabled())
  {
    msg<<"Serial ";
  }
  msg<<"\n";
 msg<<"------------------------------------------\n";
  return msg.str();
}

}
