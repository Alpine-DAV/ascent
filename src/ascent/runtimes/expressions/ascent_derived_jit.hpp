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
/// file: ascent_derived_jit.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_DERVIVED_JIT_HPP
#define ASCENT_DERVIVED_JIT_HPP

#include <ascent.hpp>
#include <conduit.hpp>
#include <flow.hpp>
#include <memory>

#include "ascent_jit_array.hpp"
#include "ascent_jit_field.hpp"
#include "ascent_jit_kernel.hpp"
#include "ascent_jit_math.hpp"
#include "ascent_jit_topology.hpp"
#include "ascent_insertion_ordered_set.hpp"
// Matt: there is a lot of code that needs its own file

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

class Jitable
{
protected:
  static int m_cuda_device_id;
public:
  Jitable(const int num_domains)
  {
    for(int i = 0; i < num_domains; ++i)
    {
      dom_info.append();
    }
    arrays.resize(num_domains);
  }

  static void init_occa();
  static void set_cuda_device(int device_id);
  static int num_cuda_devices();


  void fuse_vars(const Jitable &from);
  bool can_execute() const;
  void execute(conduit::Node &dataset, const std::string &field_name);
  std::string generate_kernel(const int dom_idx,
                              const conduit::Node &args) const;

  // map of kernel types (e.g. for different topologies)
  std::unordered_map<std::string, Kernel> kernels;
  // stores entries and argument values for each domain
  conduit::Node dom_info;
  // Store the array schemas. Used by code generation. We will copy to these
  // schemas when we execute
  std::vector<ArrayCode> arrays;
  std::string topology;
  std::string association;
  // metadata used to make the . operator work and store various jitable state
  conduit::Node obj;
};

class MemoryRegion
{
public:
  MemoryRegion(const void *start, const void *end);
  MemoryRegion(const void *start, const size_t size);
  bool operator<(const MemoryRegion &other) const;

  const unsigned char *start;
  const unsigned char *end;
  mutable bool allocated;
  mutable size_t index;
};

class JitExecutionPolicy
{
public:
  JitExecutionPolicy();
  virtual bool should_execute(const Jitable &jitable) const = 0;
  virtual std::string get_name() const = 0;
};

class FusePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class AlwaysExecutePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class RoundtripPolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

// fuse until the number of bytes in args exceeds a threshold
class InputBytesPolicy final : public JitExecutionPolicy
{
public:
  InputBytesPolicy(const size_t num_bytes);
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;

private:
  const size_t num_bytes;
};

void pack_topology(const std::string &topo_name,
                   const conduit::Node &domain,
                   conduit::Node &args,
                   ArrayCode &array);
void pack_array(const conduit::Node &array,
                const std::string &name,
                conduit::Node &args,
                ArrayCode &array_code);
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
