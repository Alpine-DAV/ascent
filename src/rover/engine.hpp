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
#ifndef rover_engine_h
#define rover_engine_h
#include <rover_types.hpp>
#include <vtkm_typedefs.hpp>

#include <vtkm/cont/ColorTable.hxx>
namespace rover {

class Engine
{
public:
  Engine(){};
  virtual ~Engine(){};

  virtual void set_data_set(vtkmDataSet &) = 0;
  virtual PartialVector32 partial_trace(Ray32 &rays) = 0;
  virtual PartialVector64 partial_trace(Ray64 &rays) = 0;
  virtual void init_rays(Ray32 &rays) = 0;
  virtual void init_rays(Ray64 &rays) = 0;
  virtual void set_primary_range(const vtkmRange &range) = 0;
  virtual void set_composite_background(bool on) = 0;
  virtual vtkmRange get_primary_range() = 0;
  virtual int get_num_channels() = 0;

  virtual void set_primary_field(const std::string &primary_field) = 0;

  virtual void set_samples(const vtkm::Bounds &global_bounds, const int &samples)
  {
    (void)samples;
    (void)global_bounds;
  }

  virtual void set_secondary_field(const std::string &secondary_field)
  {
    m_secondary_field = secondary_field;
  }

  virtual void set_color_table(const vtkmColorTable &color_map, int samples = 1024)
  {
    constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::UInt8, 4>> temp;

    color_map.Sample(samples, temp);
    m_color_map.Allocate(samples);
    auto portal = m_color_map.GetPortalControl();
    auto colorPortal = temp.GetPortalConstControl();

    for (vtkm::Id i = 0; i < samples; ++i)
    {
      auto color = colorPortal.Get(i);
      vtkm::Vec<vtkm::Float32, 4> t(color[0] * conversionToFloatSpace,
                                    color[1] * conversionToFloatSpace,
                                    color[2] * conversionToFloatSpace,
                                    color[3] * conversionToFloatSpace);
      portal.Set(i, t);
    }

  }

  void set_color_map(const vtkmColorMap &color_map)
  {
    m_color_map = color_map;
  }

  vtkmColorMap get_color_map() const
  {
    return m_color_map;
  }
protected:
  vtkmColorMap m_color_map;
  std::string m_primary_field;
  std::string m_secondary_field;
};

}; // namespace rover
#endif
