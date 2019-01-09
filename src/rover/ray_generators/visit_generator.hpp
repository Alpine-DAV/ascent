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
#ifndef rover_visit_generator_h
#define rover_visit_generator_h

#include <ray_generators/ray_generator.hpp>

namespace rover {

class VisitGenerator : public RayGenerator
{
public:
  struct VisitParams
  {
    vtkm::Vec<double,3> m_normal;
    vtkm::Vec<double,3> m_focus;
    vtkm::Vec<double,3> m_view_up;

    vtkm::Vec<double,2> m_image_pan;

    double                 m_view_angle;
    double                 m_parallel_scale;
    double                 m_near_plane;
    double                 m_far_plane;
    double                 m_image_zoom;
    bool                   m_perspective;

    VisitParams()
      : m_normal(0.f, 0.f, 0.f),
        m_focus(0.f, 0.f, 0.f),
        m_view_up(0.f, 1.f, 0.f),
        m_image_pan(0.f, 0.f),
        m_view_angle(30.f),
        m_parallel_scale(.5f),
        m_near_plane(-0.5f),
        m_far_plane(0.5f),
        m_image_zoom(1.f),
        m_perspective(true)
    { }

    void print() const
    {
      std::cout<<"******** VisIt Parmas *********\n";
      std::cout<<"normal        : "<<m_normal<<"\n";
      std::cout<<"focus         : "<<m_focus<<"\n";
      std::cout<<"up            : "<<m_view_up<<"\n";
      std::cout<<"pan           : "<<m_image_pan<<"\n";
      std::cout<<"view angle    : "<<m_view_angle<<"\n";
      std::cout<<"parallel scale: "<<m_parallel_scale<<"\n";
      std::cout<<"near_plane    : "<<m_near_plane<<"\n";
      std::cout<<"far_plane     : "<<m_far_plane<<"\n";
      std::cout<<"zoom          : "<<m_image_zoom<<"\n";
      std::cout<<"perspective   : "<<m_perspective<<"\n";
    }


  };

  VisitGenerator(const VisitParams &params);
  virtual ~VisitGenerator();

  virtual void get_rays(vtkmRayTracing::Ray<vtkm::Float32> &rays);
  virtual void get_rays(vtkmRayTracing::Ray<vtkm::Float64> &rays);

  void set_params(const VisitParams &params);
  void print_params() const;
protected:
  VisitGenerator();
  VisitParams m_params;
  template<typename T> void gen_rays(vtkmRayTracing::Ray<T> &rays);
};

} // namespace rover
#endif
