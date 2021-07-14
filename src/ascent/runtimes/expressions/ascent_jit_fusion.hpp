//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
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

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_fusion.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_JIT_FUSION_HPP
#define ASCENT_JIT_FUSION_HPP

#include <unordered_map>
#include <string>
#include <conduit.hpp>
#include "ascent_derived_jit.hpp"

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

// handles kernel fusion for various functions
// calls TopologyCode, FieldCode, etc.
// i.e., this class knows how to combine kernels and generate jitable functions
class JitableFusion
{
public:
  JitableFusion(const conduit::Node &params,
                   const std::vector<const Jitable *> &input_jitables,
                   const std::vector<const Kernel *> &input_kernels,
                   const std::string &filter_name,
                   const conduit::Node &dataset,
                   const int dom_idx,
                   const bool not_fused,
                   Jitable &out_jitable,
                   Kernel &out_kernel);

  void binary_op();
  void builtin_functions(const std::string &function_name);
  void expr_dot();
  void expr_if();
  void derived_field();
  void gradient();
  void curl();
  void recenter();
  void magnitude();
  void vector();
  void binning_value(const conduit::Node &binning);
  void rand();

private:
  void topo_attrs(const conduit::Node &obj, const std::string &name);
  void gradient(const int field_port, const int component);
  void temporary_field(const Kernel &field_kernel,
                       const std::string &field_name);
  std::string possible_temporary(const int field_port);

  const conduit::Node &params;
  const std::vector<const Jitable *> &input_jitables;
  const std::vector<const Kernel *> &input_kernels;
  const std::string &filter_name;
  const conduit::Node &dataset;
  const int dom_idx;
  const bool not_fused;
  Jitable &out_jitable;
  Kernel &out_kernel;
  const conduit::Node &inputs;
  const conduit::Node &domain;
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
