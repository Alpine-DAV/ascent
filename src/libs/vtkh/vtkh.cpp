#include "vtkh.hpp"
#include "Error.hpp"
#include <vtkh/Logger.hpp>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/RuntimeDeviceInformation.h>
#include <viskores/cont/RuntimeDeviceTracker.h>


#if defined(VISKORES_CUDA) || defined(KOKKOS_ENABLE_CUDA)
#include <cuda.h>
#endif

#ifdef KOKKOS_ENABLE_HIP
#include <hip.h>
#endif


#include <sstream>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

#ifdef VISKORES_ENABLE_KOKKOS
#include<Kokkos_Core.hpp>
#endif

namespace vtkh
{

static int  g_mpi_comm_id = -1;
static bool g_viskores_inited = false;
static bool g_vtkh_inited_kokkos = false;


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
#ifdef VTKH_PARALLEL // mpi case
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
CheckCommHandle()
{
  if(g_mpi_comm_id == -1)
  {
    std::stringstream msg;
    msg<<"VTK-h internal error. There is no valid MPI comm available. ";
    msg<<"It is likely that VTKH.SetMPICommHandle(int) was not called.";
    throw Error(msg.str());
  }
}

//---------------------------------------------------------------------------//
void
SetMPICommHandle(int mpi_comm_id)
{
  g_mpi_comm_id = mpi_comm_id;
#ifdef VTKH_ENABLE_LOGGING
  DataLogger::GetInstance()->SetRank(GetMPIRank());
#endif
}

//---------------------------------------------------------------------------//
int
GetMPICommHandle()
{
  CheckCommHandle();
  return g_mpi_comm_id;
}

//---------------------------------------------------------------------------//
int
GetMPIRank()
{
  int rank;
  MPI_Comm comm = MPI_Comm_f2c(GetMPICommHandle());
  MPI_Comm_rank(comm, &rank);
  return rank;
}

//---------------------------------------------------------------------------//
int
GetMPISize()
{
  int size;
  MPI_Comm comm = MPI_Comm_f2c(GetMPICommHandle());
  MPI_Comm_size(comm, &size);
  return size;
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
#else // non-mpi case
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
CheckCommHandle()
{
  std::stringstream msg;
  msg<<"VTK-h internal error. Trying to access MPI comm in non-mpi vtkh lib.";
  msg<<"Are you using the right library (vtkh vs vtkh_mpi)?";
  throw Error(msg.str());
}

//---------------------------------------------------------------------------//
void
SetMPICommHandle(int mpi_comm_id)
{
  std::stringstream msg;
  msg<<"VTK-h internal error. Trying to set MPI comm handle in non-mpi vtkh lib.";
  msg<<"Are you using the right library (vtkh vs vtkh_mpi)?";
  throw Error(msg.str());
}

//---------------------------------------------------------------------------//
int
GetMPICommHandle()
{
  std::stringstream msg;
  msg<<"VTK-h internal error. Trying to get MPI comm handle in non-mpi vtkh lib.";
  msg<<"Are you using the right library (vtkh vs vtkh_mpi)?";
  throw Error(msg.str());
  return g_mpi_comm_id;
}

//---------------------------------------------------------------------------//
int
GetMPIRank()
{
  return 0;
}

//---------------------------------------------------------------------------//
int
GetMPISize()
{
  return 1;
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Initialize()
{
    // call viskores init, if we haven't already
    if(!g_viskores_inited)
    {
        viskores::cont::Initialize();
        g_viskores_inited = true;
    }
}

//---------------------------------------------------------------------------//
bool
IsMPIEnabled()
{
#ifdef VTKH_PARALLEL
  return true;
#else
  return false;
#endif
}

std::string GetCurrentDevice()
{
  std::string device = "serial";
  // use the same prefered ordering as viskores
  if(IsCUDAEnabled())
  {
    device = "cuda";
  }
  else if(IsOpenMPEnabled())
  {
    device = "openmp";
  }
  else if(IsKokkosEnabled())
  {
    device = "kokkos";
  }

  return device;
}

//---------------------------------------------------------------------------//
bool
IsSerialAvailable()
{
  viskores::cont::RuntimeDeviceInformation info;
  return info.Exists(viskores::cont::DeviceAdapterTagSerial());
}


//---------------------------------------------------------------------------//
bool
IsOpenMPAvailable()
{
  viskores::cont::RuntimeDeviceInformation info;
  return info.Exists(viskores::cont::DeviceAdapterTagOpenMP());
}

//---------------------------------------------------------------------------//
bool
IsCUDAAvailable()
{
  viskores::cont::RuntimeDeviceInformation info;
  return info.Exists(viskores::cont::DeviceAdapterTagCuda());
}

//---------------------------------------------------------------------------//
bool
IsKokkosAvailable()
{
  viskores::cont::RuntimeDeviceInformation info;
  return info.Exists(viskores::cont::DeviceAdapterTagKokkos());
}

//---------------------------------------------------------------------------//
bool
IsSerialEnabled()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  return device_tracker.CanRunOn(viskores::cont::DeviceAdapterTagSerial());
}


//---------------------------------------------------------------------------//
bool
IsOpenMPEnabled()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  return device_tracker.CanRunOn(viskores::cont::DeviceAdapterTagOpenMP());
}

//---------------------------------------------------------------------------//
bool
IsCUDAEnabled()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  return device_tracker.CanRunOn(viskores::cont::DeviceAdapterTagCuda());
}

//---------------------------------------------------------------------------//
bool
IsKokkosEnabled()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  return device_tracker.CanRunOn(viskores::cont::DeviceAdapterTagKokkos());
}

//---------------------------------------------------------------------------//
int
CUDADeviceCount()
{
    int device_count = 0;
#ifdef VISKORES_CUDA
    cudaError_t res = cudaGetDeviceCount(&device_count);
    if(res != cudaSuccess)
    {
        std::stringstream msg;
        msg << "Failed to get CUDA device count" << std::endl
            << "CUDA Error Message: "
            << cudaGetErrorString(res);
        throw Error(msg.str());
    }
    return device_count;

#else
    throw Error("Cannot fetch CUDA device count: Viskores lacks CUDA support");
#endif
    return device_count;
}

//---------------------------------------------------------------------------//
void
SelectCUDADevice(int device_index)
{
#ifdef VISKORES_CUDA
    int device_count = CUDADeviceCount();
    // make sure index is ok
    if(device_index >= 0 && device_index < device_count)
    {
        cudaError_t res = cudaSetDevice(device_index);
        if(res != cudaSuccess)
        {
            std::stringstream msg;
            msg << "Failed to set CUDA device (device index = "
                << device_index << ")" << std::endl
                << "CUDA Error Message: "
                << cudaGetErrorString(res);
            throw Error(msg.str());
        }
    }
    else
    {
        std::stringstream msg;
        msg << "Invalid CUDA device index: "
            << device_index
            << " (number of devices = "
            << device_index << ")";
        throw Error(msg.str());
    }
#else
    throw Error("Cannot set CUDA device: Viskores lacks CUDA support");
#endif

}
//---------------------------------------------------------------------------//
void
InitializeKokkos()
{
  SelectKokkosDevice(0);
}

//---------------------------------------------------------------------------//
int
KokkosDeviceCount()
{
    int device_count = 0;
#ifdef VISKORES_ENABLE_KOKKOS
    // NEW KOKKOS API makes this easier, use it when we have access
    // device_count = Kokkos::num_devices();

    #ifdef KOKKOS_ENABLE_HIP
    // kokkos + hip case
    {
        hipError_t res = hipGetDeviceCount(&device_count);
        if(res != hipSuccess)
        {
            std::stringstream msg;
            msg << "Failed to get HIP device count" << std::endl
                << "HIP Error Message: "
                << hipGetErrorString(res);
            throw Error(msg.str());
        }
    }
    #endif

    #ifdef KOKKOS_ENABLE_CUDA
        // kokkos + cuda case
        {
            cudaError_t res = cudaGetDeviceCount(&device_count);
            if(res != cudaSuccess)
            {
                std::stringstream msg;
                msg << "Failed to get CUDA device count" << std::endl
                    << "CUDA Error Message: "
                    << cudaGetErrorString(res);
                throw Error(msg.str());
            }
        }
    #endif
#else
    throw Error("Cannot fetch Kokkos device count: Viskores lacks Kokkos support");
#endif
    return device_count;
}


//---------------------------------------------------------------------------//
void
SelectKokkosDevice(int device_index)
{ 
#ifdef VISKORES_ENABLE_KOKKOS
    // if kokkos is not already inited
    if(!Kokkos::is_initialized())
    {
        // only set if we have devices
        if(KokkosDeviceCount() > 0 )
        {
            // TODO: is this newer kokkos api than we are using?
            Kokkos::InitializationSettings settings;
            // If Kokkos was built with CUDA or HIP enabled, use the GPU with device ID.
            settings.set_device_id(device_index);
        }
        // init Kokkos
        Kokkos::initialize();
        g_vtkh_inited_kokkos = true;
    }
#endif
}

//---------------------------------------------------------------------------//
void
ForceSerial()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagSerial());
}

//---------------------------------------------------------------------------//
void
ForceOpenMP()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagOpenMP());
}

//---------------------------------------------------------------------------//
void
ForceCUDA()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda());
}

//---------------------------------------------------------------------------//
void
ForceKokkos()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagKokkos());
}

//---------------------------------------------------------------------------//
void
ResetDevices()
{
  viskores::cont::RuntimeDeviceTracker &device_tracker
    = viskores::cont::GetRuntimeDeviceTracker();
  device_tracker.Reset();
}

//---------------------------------------------------------------------------//
std::string
AboutVTKH()
{
  std::stringstream msg;
  msg<<"---------------- VTK-h -------------------\n";
#ifdef VTKH_PARALLEL
  int version, subversion;
  MPI_Get_version(&version, &subversion);
  msg<<"MPI version: "<<version<<"."<<subversion<<"\n";
#else
  msg<<"MPI version: n/a\n";
#endif
  msg<<"Viskores adapters: ";

  if(IsCUDAAvailable())
  {
    msg<<"Cuda (";
    if(IsCUDAEnabled())
    {
      msg<<"enabled) ";
    }
    else
    {
      msg<<"disabled) ";
    }

  }

  if(IsOpenMPAvailable())
  {
    msg<<"OpenMP (";
    if(IsOpenMPEnabled())
    {
      msg<<"enabled) ";
    }
    else
    {
      msg<<"disabled) ";
    }
  }

  if(IsKokkosAvailable())
  {
    msg<<"Kokkos (";
    if(IsKokkosEnabled())
    {
      msg<<"enabled) ";
    }
    else
    {
      msg<<"disabled) ";
    }
  }

  if(IsSerialAvailable())
  {
    msg<<"Serial (";
    if(IsSerialEnabled())
    {
      msg<<"enabled) ";
    }
    else
    {
      msg<<"disabled) ";
    }
  }
  msg<<"\n";
 msg<<"------------------------------------------\n";
  return msg.str();
}

}
