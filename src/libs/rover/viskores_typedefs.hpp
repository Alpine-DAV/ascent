//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_viskores_typedefs_h
#define rover_viskores_typedefs_h

#include <rover_config.h>

#include <vtkh/rendering/PartialComposite.hpp>

#include <viskores/cont/DataSet.h>
#include <viskores/cont/Timer.h>
#include <viskores/cont/ColorTable.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/raytracing/Camera.h>
#include <viskores/rendering/raytracing/Ray.h>
#include <viskores/rendering/raytracing/Logger.h>

namespace rover {


// vtkh
namespace vtkhRayTracing = vtkh::rendering::raytracing;
typedef std::vector<vtkh::rendering::raytracing::PartialComposite<viskores::Float32>> PartialVector32;
typedef std::vector<vtkh::rendering::raytracing::PartialComposite<viskores::Float64>> PartialVector64;

// viskores
namespace viskoresRayTracing = viskores::rendering::raytracing;
typedef viskores::Range                                           viskoresRange;
typedef viskores::cont::DataSet                                   viskoresDataSet;
typedef viskores::cont::CoordinateSystem                          viskoresCoordinates;
typedef viskores::rendering::raytracing::Ray<viskores::Float32>       Ray32;
typedef viskores::rendering::raytracing::Ray<viskores::Float64>       Ray64;
typedef viskores::cont::ColorTable                                viskoresColorTable;
typedef viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32, 4>> viskoresColorMap;
typedef viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32, 4>> viskoresColorBuffer;
typedef viskores::rendering::Camera                               viskoresCamera;
typedef viskores::rendering::raytracing::Camera                   viskoresRayCamera;
typedef viskores::cont::ArrayHandle<viskores::Id>                     IdHandle;
typedef viskores::Vec<viskores::Float32, 2>                           viskoresVec2f;
typedef viskores::Vec<viskores::Float32, 3>                           viskoresVec3f;
typedef viskores::cont::Timer                                     viskoresTimer;
typedef viskores::rendering::raytracing::Logger                   viskoresLogger;

//
// Utility method for getting raw pointer
//
template<typename T>
T *
get_viskores_ptr(viskores::cont::ArrayHandle<T> handle)
{
  return handle.WritePortal().GetArray();
}

};
#endif
