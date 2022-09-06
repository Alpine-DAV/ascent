#ifndef _ADAPTER_H_
#define _ADAPTER_H_

#if 1

#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_ERROR

#include <vtkm/cont/serial/DeviceAdapterSerial.h>
#define IO_DEVICE_ADAPTER vtkm::cont::DeviceAdapterTagSerial
#ifdef USE_TBB
#include <vtkm/cont/tbb/DeviceAdapterTBB.h>
#include <tbb/task_scheduler_init.h>
#define WORKER_DEVICE_ADAPTER vtkm::cont::DeviceAdapterTagTBB
#elif USE_CUDA
#include <vtkm/cont/cuda/DeviceAdapterCuda.h>
#define WORKER_DEVICE_ADAPTER vtkm::cont::DeviceAdapterTagCuda
#else
#define WORKER_DEVICE_ADAPTER vtkm::cont::DeviceAdapterTagSerial
#endif

#endif

#endif
