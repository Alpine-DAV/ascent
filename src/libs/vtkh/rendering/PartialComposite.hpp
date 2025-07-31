//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

//
// This code was originally pulled in from Viskores.
//

#ifndef vtkh_rendering_raytracing_PartialComposite_h
#define vtkh_rendering_raytracing_PartialComposite_h

#include <vtkh/vtkh_exports.h>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/rendering/raytracing/ChannelBuffer.h>

namespace vtkh
{
namespace rendering
{
namespace raytracing
{

template <typename FloatType>
struct VTKH_API PartialComposite
{
  vtkm::cont::ArrayHandle<vtkm::Id> PixelIds;   // pixel that owns composite
  vtkm::cont::ArrayHandle<FloatType> Distances; // distance of composite end
  vtkm::rendering::raytracing::ChannelBuffer<FloatType> Transmission;              // holds either color or absorption
  vtkm::rendering::raytracing::ChannelBuffer<FloatType> Intensity;           // holds the intensity emerging from each ray
  vtkm::rendering::raytracing::ChannelBuffer<FloatType> OpticalDepth;
  vtkm::cont::ArrayHandle<FloatType> PathLengths; // Total distance traversed through the mesh
};
}
}
} // namespace vtkm::rendering::raytracing
#endif
