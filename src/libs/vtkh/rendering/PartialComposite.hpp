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

#include <viskores/cont/ArrayHandle.h>
#include <viskores/rendering/raytracing/ChannelBuffer.h>

namespace vtkh
{
namespace rendering
{
namespace raytracing
{

template <typename FloatType>
struct VTKH_API PartialComposite
{
  viskores::cont::ArrayHandle<viskores::Id> PixelIds;   // pixel that owns composite
  viskores::cont::ArrayHandle<FloatType> Distances; // distance of composite end
  viskores::rendering::raytracing::ChannelBuffer<FloatType> Transmission;              // holds either color or absorption
  viskores::rendering::raytracing::ChannelBuffer<FloatType> Intensity;           // holds the intensity emerging from each ray
  viskores::rendering::raytracing::ChannelBuffer<FloatType> OpticalDepth;
  viskores::cont::ArrayHandle<FloatType> PathLengths; // Total distance traversed through the mesh
};
}
}
} // namespace viskores::rendering::raytracing
#endif
