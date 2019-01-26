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
#ifndef rover_image_h
#define rover_image_h

#include <vector>

#include <vtkm/cont/ArrayHandle.h>

#include <rover_types.hpp>
#include <partial_image.hpp>

namespace rover
{

template<typename FloatType>
class Image
{
public:
  typedef vtkm::cont::ArrayHandle<FloatType> HandleType;

  Image();
  Image(PartialImage<FloatType> &partial);

  FloatType * steal_intensity(const int &channel_num);
  HandleType  get_intensity(const int &channel_num);
  FloatType * steal_optical_depth(const int &channel_num);
  HandleType  get_optical_depth(const int &channel_num);
  int get_num_channels() const;
  bool has_intensity(const int &channel_num) const;
  bool has_optical_depth(const int &channel_num) const;
  void normalize_intensity(const int &channel_num);
  void normalize_optical_depth(const int &channel_num);
  void operator=(PartialImage<FloatType> partial);
  template<typename O> void operator=(Image<O> &other);
  HandleType flatten_intensities();
  HandleType flatten_optical_depths();
  int get_size();
  template<typename T,
           typename O> friend void init_from_image(Image<T> &left,
                                                   Image<O> &right);

protected:
  int                                      m_height;
  int                                      m_width;
  std::vector<HandleType>                  m_intensities;
  std::vector<HandleType>                  m_optical_depths;
  std::vector<bool>                        m_valid_intensities;
  std::vector<bool>                        m_valid_optical_depths;

  void init_from_partial(PartialImage<FloatType> &);
  void normalize_handle(HandleType &, bool);
};
} // namespace rover
#endif
