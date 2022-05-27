#ifndef VTK_H_HPP
#define VTK_H_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh_config.h>
#include <string>

namespace vtkh
{

  VTKH_API std::string AboutVTKH();
  VTKH_API void        Initialize();

  // is backend support compiled in
  VTKH_API bool        IsSerialAvailable();
  VTKH_API bool        IsOpenMPAvailable();
  VTKH_API bool        IsCUDAAvailable();
  VTKH_API bool        IsKokkosAvailable();

  // is backend enabled (e.g., ForceX)
  VTKH_API bool        IsSerialEnabled();
  VTKH_API bool        IsOpenMPEnabled();
  VTKH_API bool        IsCUDAEnabled();
  VTKH_API bool        IsKokkosEnabled();

  VTKH_API bool        IsMPIEnabled();

  VTKH_API int         CUDADeviceCount();
  VTKH_API void        SelectCUDADevice(int device_index);
  VTKH_API void        SelectKokkosDevice(int device_index);

  VTKH_API void        ForceSerial();
  VTKH_API void        ForceOpenMP();
  VTKH_API void        ForceCUDA();
  VTKH_API void        ForceKokkos();
  VTKH_API void        ResetDevices();
  VTKH_API std::string GetCurrentDevice();

  VTKH_API int         GetMPIRank();
  VTKH_API int         GetMPISize();

  VTKH_API void        SetMPICommHandle(int mpi_comm_id);
  VTKH_API int         GetMPICommHandle();
}
#endif
