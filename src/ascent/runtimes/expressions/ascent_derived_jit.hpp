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
#include <unordered_map>
#include <unordered_set>

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
void pack_topo(const std::string &topo_name,
               const conduit::Node &dom,
               conduit::Node &args);

template <typename T>
class InsertionOrderedSet
{
public:
  void
  insert(const T &item)
  {
    if(data_set.find(item) == data_set.end())
    {
      data_set.insert(item);
      insertion_ordered_data.push_back(item);
    }
  }

  void
  insert(std::initializer_list<T> ilist)
  {
    for(const auto &item : ilist)
    {
      insert(item);
    }
  }
  void
  insert(const InsertionOrderedSet<T> &ios)
  {
    for(const auto &item : ios.data())
    {
      insert(item);
    }
  }

  const std::vector<T> &
  data() const
  {
    return insertion_ordered_data;
  }

private:
  std::unordered_set<T> data_set;
  std::vector<T> insertion_ordered_data;
};

class TopologyCode
{
public:
  TopologyCode(const std::string &topo_name, const conduit::Node &dom);
  void vertex_idx(InsertionOrderedSet<std::string> &code);
  void vertex_xyz(InsertionOrderedSet<std::string> &code);
  void cell_idx(InsertionOrderedSet<std::string> &code);
  void cell_xyz(InsertionOrderedSet<std::string> &code);
  void dxdydz(InsertionOrderedSet<std::string> &code);
  void volume(InsertionOrderedSet<std::string> &code);

private:
  int num_dims;
  std::string topo_type;
  std::string topo_name;
};

class Kernel
{
public:
  void fuse_kernel(const Kernel &from);
  std::string generate_for_body(const std::string &output,
                                bool output_exists) const;
  std::string generate_loop(const std::string &output) const;

  std::string kernel_body;
  std::string for_body;
  InsertionOrderedSet<std::string> inner_scope;
  std::string expr;
  conduit::Node obj;

private:
  std::string generate_inner_scope() const;
};

class Jitable
{
public:
  Jitable(const int num_domains)
  {
    for(int i = 0; i < num_domains; ++i)
    {
      dom_info.append();
    }
  }

  void fuse_vars(const Jitable &from);
  void execute(conduit::Node &dataset, const std::string &field_name);
  std::string generate_kernel(const int dom_idx) const;

  std::unordered_map<std::string, Kernel> kernels;
  conduit::Node dom_info;
  std::string topology;
  std::string association;
};
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
