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
#ifndef rover_scheduler_h
#define rover_scheduler_h

#include <domain.hpp>
#include <image.hpp>
#include <engine.hpp>
#include <scheduler_base.hpp>
#include <rover_types.hpp>
#include <ray_generators/ray_generator.hpp>
#include <vtkm_typedefs.hpp>

//
// Scheduler types:
//  static: all ranks gets all rays
//  normal compositing -
//    back to front (energy): absorbtion, absorbtion + emmission
//    front to back (volume): normal volume rendering
//  dynamic(scattering):
//    domain passing -
//      front to back: volume rendering and ray tracing
//      back to front: both energy types.
//
//
//
namespace rover {

template<typename FloatType>
class Scheduler : public SchedulerBase
{
public:
  Scheduler();
  virtual ~Scheduler();
  void trace_rays() override;
  void save_result(std::string file_name) override;
  void save_result(std::string file_name,
                   float min_val,
                   float max_val,
                   bool log_scale) override;
  void save_bov(std::string file_name) override;
  virtual void to_blueprint(conduit::Node &dataset) override;

  virtual void get_result(Image<vtkm::Float32> &image) override;
  virtual void get_result(Image<vtkm::Float64> &image) override;
protected:
  void composite();
  void set_global_scalar_range();
  void set_global_bounds();
  int  get_global_channels();
  Image<FloatType>                          m_result;
  std::vector<PartialImage<FloatType>>      m_partial_images;

  void add_partial(vtkmRayTracing::PartialComposite<FloatType> &partial, int width, int height);
private:

};

}; // namespace rover
#endif
