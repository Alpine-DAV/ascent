//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-749865
//
// All rights reserved.
//
// This file is part of Rover.
//
// Please also read rover/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef rover_vtkm_typedefs_h
#define rover_vtkm_typedefs_h

#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DeviceAdapterListTag.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/raytracing/Camera.h>
#include <vtkm/rendering/raytracing/PartialComposite.h>
#include <vtkm/rendering/raytracing/Ray.h>
#include <vtkm/rendering/raytracing/Logger.h>

namespace rover {
namespace vtkmRayTracing = vtkm::rendering::raytracing;
typedef vtkm::Range                                           vtkmRange;
typedef vtkm::cont::DataSet                                   vtkmDataSet;
typedef vtkm::cont::CoordinateSystem                          vtkmCoordinates;
typedef vtkm::rendering::raytracing::Ray<vtkm::Float32>       Ray32;
typedef vtkm::rendering::raytracing::Ray<vtkm::Float64>       Ray64;
typedef vtkm::cont::ColorTable                                vtkmColorTable;
typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4> > vtkmColorMap;
typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4> > vtkmColorBuffer;
//typedef vtkm::rendering::raytracing::Camera                   vtkmCamera;
typedef vtkm::rendering::Camera                               vtkmCamera;
typedef vtkm::cont::ArrayHandle<vtkm::Id>                     IdHandle;
typedef vtkm::Vec<vtkm::Float32,3>                            vtkmVec3f;
typedef vtkm::cont::Timer<>                                   vtkmTimer;
typedef vtkm::rendering::raytracing::Logger                   vtkmLogger;

using PartialVector64 = std::vector<vtkm::rendering::raytracing::PartialComposite<vtkm::Float64>>;
using PartialVector32 = std::vector<vtkm::rendering::raytracing::PartialComposite<vtkm::Float32>>;

//
// Utility method for getting raw pointer
//
template<typename T>
T *
get_vtkm_ptr(vtkm::cont::ArrayHandle<T> handle)
{
  typedef typename vtkm::cont::ArrayHandle<T>  HandleType;
  typedef typename HandleType::template ExecutionTypes<vtkm::cont::DeviceAdapterTagSerial>::Portal PortalType;
  typedef typename vtkm::cont::ArrayPortalToIterators<PortalType>::IteratorType IteratorType;

  IteratorType iter = vtkm::cont::ArrayPortalToIterators<PortalType>(handle.GetPortalControl()).GetBegin();
  return &(*iter);
}

};
#endif
