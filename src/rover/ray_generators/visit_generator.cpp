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
#include <ray_generators/visit_generator.hpp>
#include <utils/rover_logging.hpp>
#include <vtkm/VectorAnalysis.h>
#include <assert.h>
#include <limits>
namespace rover {

VisitGenerator::VisitGenerator()
 : RayGenerator()
{

}

VisitGenerator::VisitGenerator(const VisitParams &params)
 : RayGenerator()
{
  m_params = params;
  assert(m_width > 0);
  assert(m_height > 0);
}

VisitGenerator::~VisitGenerator()
{

}

template<typename T>
void 
VisitGenerator::gen_rays(vtkmRayTracing::Ray<T> &rays) 
{
  vtkmTimer timer;
  double time = 0;
  ROVER_DATA_OPEN("visit_ray_gen");

  const int size = m_width * m_height;

  rays.Resize(size, vtkm::cont::DeviceAdapterTagSerial());
  
  vtkm::Vec<T,3> view_side;

  view_side[0] = m_params.m_view_up[1] * m_params.m_normal[2] 
                 - m_params.m_view_up[2] * m_params.m_normal[1];

  view_side[1] = -m_params.m_view_up[0] * m_params.m_normal[2] 
                 + m_params.m_view_up[2] * m_params.m_normal[0];

  view_side[2] = m_params.m_view_up[0] * m_params.m_normal[1] 
                 - m_params.m_view_up[1] * m_params.m_normal[0];

  T near_height, view_height, far_height;
  T near_width, view_width, far_width;;

  view_height = m_params.m_parallel_scale;
  // I think this is flipped
  view_width = view_height * (m_height / m_width);
  if(m_params.m_perspective)
  {
    T view_dist = m_params.m_parallel_scale / tan((m_params.m_view_angle * 3.1415926535) / 360.);
    T near_dist = view_dist + m_params.m_near_plane;
    T far_dist  = view_dist + m_params.m_far_plane;
    near_height = (near_dist * view_height) / view_dist;
    near_width  = (near_dist * view_width) / view_dist;
    far_height  = (far_dist * view_height) / view_dist;
    far_width   = (far_dist * view_width) / view_dist;
  }
  else
  {
    near_height = view_height;
    near_width  = view_width;
    far_height  = view_height;
    far_width   = view_width;
  }

  near_height = near_height / m_params.m_image_zoom;
  near_width  = near_width  / m_params.m_image_zoom;
  far_height  = far_height  / m_params.m_image_zoom;
  far_width   = far_width   / m_params.m_image_zoom;

  vtkm::Vec<T,3> near_origin;
  vtkm::Vec<T,3> far_origin;
  near_origin = m_params.m_focus + m_params.m_near_plane * m_params.m_normal;
  far_origin = m_params.m_focus + m_params.m_far_plane * m_params.m_normal;

  T near_dx, near_dy, far_dx, far_dy;
  near_dx = (2. * near_width)  / m_width;
  near_dy = (2. * near_height) / m_height;
  far_dx  = (2. * far_width)   / m_width;
  far_dy  = (2. * far_height)  / m_height;

  auto origin_x = rays.OriginX.GetPortalControl(); 
  auto origin_y = rays.OriginY.GetPortalControl(); 
  auto origin_z = rays.OriginZ.GetPortalControl(); 

  auto dir_x = rays.DirX.GetPortalControl(); 
  auto dir_y = rays.DirY.GetPortalControl(); 
  auto dir_z = rays.DirZ.GetPortalControl(); 

  auto pixel_id = rays.PixelIdx.GetPortalControl(); 
  const int x_size = m_width; 
  const int y_size = m_height; 

  const T x_factor = - (2. * m_params.m_image_pan[0] * m_params.m_image_zoom + 1.);
  const T x_start  = x_factor * near_width + near_dx / 2.;
  const T x_end    = x_factor * far_width + far_dx / 2.;

  const T y_factor = - (2. * m_params.m_image_pan[1] * m_params.m_image_zoom + 1.);
  const T y_start  = y_factor * near_height + near_dy / 2.;
  const T y_end    = y_factor * far_height + far_dy / 2.;

  for(int y = 0; y < y_size; ++y)
  {
    const T near_y = y_start + T(y) * near_dy;
    const T far_y = y_end + T(y) * far_dy;
    #pragma omp parallel for
    for(int x = 0; x < x_size; ++x)
    {
      const int id = y * x_size + x;    

      T near_x = x_start + T(x) * near_dx;
      T far_x = x_end + T(x) * far_dx;

      vtkm::Vec<T,3> start;
      vtkm::Vec<T,3> end;
      start = near_origin + near_x * view_side + near_y * m_params.m_view_up;
      end = far_origin + far_x * view_side + far_y * m_params.m_view_up;

      vtkm::Vec<T,3> dir = end - start;
      vtkm::Normalize(dir);

      pixel_id.Set(id, id);
      origin_x.Set(id, start[0]);
      origin_y.Set(id, start[1]);
      origin_z.Set(id, start[2]);

      dir_x.Set(id, dir[0]);
      dir_y.Set(id, dir[1]);
      dir_z.Set(id, dir[2]);
      
    }
  }
 auto hit_portal = rays.HitIdx.GetPortalControl();
 auto min_portal = rays.MinDistance.GetPortalControl();
 auto max_portal = rays.MaxDistance.GetPortalControl();
  
  // set a couple other ray variables
  ROVER_INFO("Ray size "<<size);
  #pragma omp parallel for
  for(int i = 0; i < size; ++i)
  {
    hit_portal.Set(i, -2);
    min_portal.Set(i, 0.f);
    max_portal.Set(i, std::numeric_limits<T>::max());
  }
  
  time = timer.GetElapsedTime();
  ROVER_DATA_CLOSE(time);

}

void 
VisitGenerator::get_rays(vtkmRayTracing::Ray<vtkm::Float32> &rays) 
{
  gen_rays(rays);
}

void 
VisitGenerator::get_rays(vtkmRayTracing::Ray<vtkm::Float64> &rays) 
{
  gen_rays(rays);
}

void
VisitGenerator::set_params(const VisitParams &params)
{
  m_params = params;
}

void
VisitGenerator::print_params() const 
{
  m_params.print();
}


} // namespace rover
