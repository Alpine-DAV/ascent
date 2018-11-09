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
#ifndef rover_volume_block_h
#define rover_volume_block_h

#include <rover_types.hpp>
#include <limits>
namespace rover {

template<typename FloatType>
struct VolumePartial
{
  typedef FloatType ValueType;
  int                    m_pixel_id;
  float                  m_depth; 
  float                  m_pixel[3];
  float                  m_alpha;

  VolumePartial()
    : m_pixel_id(0),
      m_depth(0.f),
      m_alpha(0.f)
  {
    m_pixel[0] = 0;
    m_pixel[1] = 0;
    m_pixel[2] = 0;
  }

  void print() const
  {
    std::cout<<"[id : "<<m_pixel_id<<", red : "<<m_pixel[0]<<","
             <<" green : "<<m_pixel[1]<<", blue : "<<m_pixel[2]
             <<", alpha "<<m_alpha<<", depth : "<<m_depth<<"]\n";
  }

  bool operator < (const VolumePartial &other) const
  {
    if(m_pixel_id != other.m_pixel_id) 
    {
      return m_pixel_id < other.m_pixel_id;
    }
    else
    {
      return m_depth < other.m_depth;
    }
  }

  inline void blend(const VolumePartial &other)
  {
    if(m_alpha >= 1.f || other.m_alpha == 0.f) return;

    const float opacity = (1.f - m_alpha);
    m_pixel[0] +=  opacity * other.m_pixel[0]; 
    m_pixel[1] +=  opacity * other.m_pixel[1]; 
    m_pixel[2] +=  opacity * other.m_pixel[2]; 
    m_alpha += opacity * other.m_alpha;
    m_alpha = m_alpha > 1.f ? 1.f : m_alpha;

  }

  inline void load_from_partial(const PartialImage<ValueType> &partial_image, const int &index)
  {
    
    m_pixel_id = static_cast<int>(partial_image.m_pixel_ids.GetPortalConstControl().Get(index));
    m_depth = static_cast<float>(partial_image.m_distances.GetPortalConstControl().Get(index));

    m_pixel[0] = static_cast<float>(partial_image.m_buffer.Buffer.GetPortalConstControl().Get(index*4+0));
    m_pixel[1] = static_cast<float>(partial_image.m_buffer.Buffer.GetPortalConstControl().Get(index*4+1));
    m_pixel[2] = static_cast<float>(partial_image.m_buffer.Buffer.GetPortalConstControl().Get(index*4+2));

    m_alpha = static_cast<float>(partial_image.m_buffer.Buffer.GetPortalConstControl().Get(index*4+3));
  }
  
  inline void store_into_partial(PartialImage<FloatType> &output, 
                                 const int &index,
                                 const std::vector<FloatType> &background)
  {
    output.m_pixel_ids.GetPortalControl().Set(index, m_pixel_id ); 
    output.m_distances.GetPortalControl().Set(index, m_depth ); 
    const int starting_index = index * 4;
    output.m_buffer.Buffer.GetPortalControl().Set(starting_index + 0, static_cast<FloatType>(m_pixel[0]));
    output.m_buffer.Buffer.GetPortalControl().Set(starting_index + 1, static_cast<FloatType>(m_pixel[1]));
    output.m_buffer.Buffer.GetPortalControl().Set(starting_index + 2, static_cast<FloatType>(m_pixel[2]));
    output.m_buffer.Buffer.GetPortalControl().Set(starting_index + 3, static_cast<FloatType>(m_alpha));

    VolumePartial bg_color;
    bg_color.m_pixel[0] = static_cast<float>(background[0]);
    bg_color.m_pixel[1] = static_cast<float>(background[1]); 
    bg_color.m_pixel[2] = static_cast<float>(background[2]); 
    bg_color.m_alpha    = static_cast<float>(background[3]); 

    this->blend(bg_color);

    output.m_intensities.Buffer.GetPortalControl().Set(starting_index + 0, static_cast<FloatType>(m_pixel[0]));
    output.m_intensities.Buffer.GetPortalControl().Set(starting_index + 1, static_cast<FloatType>(m_pixel[1]));
    output.m_intensities.Buffer.GetPortalControl().Set(starting_index + 2, static_cast<FloatType>(m_pixel[2]));
    output.m_intensities.Buffer.GetPortalControl().Set(starting_index + 3, static_cast<FloatType>(m_alpha));
  }

  static void composite_background(std::vector<VolumePartial> &partials, 
                                   const std::vector<FloatType> &background)
  {
    VolumePartial bg_color;
    bg_color.m_pixel[0] = static_cast<float>(background[0]);
    bg_color.m_pixel[1] = static_cast<float>(background[1]); 
    bg_color.m_pixel[2] = static_cast<float>(background[2]); 
    bg_color.m_alpha    = static_cast<float>(background[3]); 
    //
    // Gather the unique pixels into the output
    //
    const int total_pixels = static_cast<int>(partials.size());
    #pragma omp parallel for 
    for(int i = 0; i < total_pixels; ++i)
    { 
      partials[i].blend(bg_color);
    }

  }
 
};

} // namespace
#endif
