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
#ifndef rover_compositor_h
#define rover_compositor_h

#include <vector>
#include <iostream>
#include <vtkm/Types.h>
#include <vtkh/vtkh_exports.h>
#include "AbsorptionPartial.hpp"
#include "EmissionPartial.hpp"
#include "VolumePartial.hpp"


namespace vtkh {

template<typename PartialType>
class VTKH_API PartialCompositor
{
public:
  PartialCompositor();
  ~PartialCompositor();
  void
  composite(std::vector<std::vector<PartialType>> &partial_images,
            std::vector<PartialType> &output_partials);
  void set_background(std::vector<vtkm::Float32> &background_values);
  void set_background(std::vector<vtkm::Float64> &background_values);
  void set_comm_handle(int mpi_comm_id);
protected:
  void merge(const std::vector<std::vector<PartialType>> &in_partials,
             std::vector<PartialType> &partials,
             int &global_min_pixel,
             int &global_max_pixel);

  void composite_partials(std::vector<PartialType> &partials,
                          std::vector<PartialType> &output_partials);

  std::vector<typename PartialType::ValueType> m_background_values;
  int m_mpi_comm_id;
};

}; // namespace rover
#endif
