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
#include <ray_generators/camera_generator.hpp>

namespace rover
{

CameraGenerator::CameraGenerator()
 : RayGenerator()
{

}

CameraGenerator::CameraGenerator(const vtkmCamera &camera)
 : RayGenerator()
{
  m_camera = camera;
}

CameraGenerator::~CameraGenerator()
{

}

void
CameraGenerator::get_rays(Ray32 &rays)
{
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  vtkmRayCamera ray_gen;
  ray_gen.SetParameters(m_camera, width, height);
  ray_gen.CreateRays(rays, this->m_coordinates.GetBounds());
  this->m_has_rays = false;
  if (rays.NumRays == 0)
  {
    ROVER_WARN("CameraGenerator::get_rays: No rays were generated");
  }
}

void
CameraGenerator::get_rays(Ray64 &rays)
{
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  vtkmRayCamera ray_gen;
  ray_gen.SetParameters(m_camera, width, height);
  ray_gen.CreateRays(rays, this->m_coordinates.GetBounds());
  this->m_has_rays = false;
  if (rays.NumRays == 0)
  {
    ROVER_WARN("CameraGenerator::get_rays: No rays were generated");
  }
}

void
CameraGenerator::set_camera(vtkmCamera &camera)
{
  m_camera = camera;
}

vtkmCoordinates
CameraGenerator::get_coordinates()
{
  return m_coordinates;
}

void
CameraGenerator::set_coordinates(vtkmCoordinates coordinates)
{
  m_coordinates = coordinates;
}

} // namespace rover
