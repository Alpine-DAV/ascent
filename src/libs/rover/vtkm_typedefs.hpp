//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_vtkm_typedefs_h
#define rover_vtkm_typedefs_h
#include <rover_config.h>
#include <vtkm/cont/DataSet.h>
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
typedef vtkm::cont::Timer                                     vtkmTimer;
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
  return handle.WritePortal().GetArray();
}

};
#endif
