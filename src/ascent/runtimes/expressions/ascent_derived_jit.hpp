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
// TODO maybe move Topology class into its own file
#include "ascent_blueprint_architect.hpp"

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
template <typename T>
class InsertionOrderedSet
{
public:
  void
  insert(const T &item, const bool unique = true)
  {
    if(!unique)
    {
      insertion_ordered_data.push_back(item);
    }
    else if(data_set.find(item) == data_set.end())
    {
      data_set.insert(item);
      insertion_ordered_data.push_back(item);
    }
  }

  void
  insert(std::initializer_list<T> ilist, const bool unique = true)
  {
    for(const auto &item : ilist)
    {
      insert(item, unique);
    }
  }

  void
  insert(const InsertionOrderedSet<T> &ios, const bool unique = true)
  {
    for(const auto &item : ios.data())
    {
      insert(item, unique);
    }
  }

  T
  accumulate() const
  {
    T res;
    for(const auto &item : insertion_ordered_data)
    {
      res += item;
    }
    return res;
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

class MathCode
{
public:
  void determinant_2x2(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const bool declare = true);
  void determinant_3x3(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &c,
                       const std::string &res_name,
                       const bool declare = true);
  void vector_subtract(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const int num_components,
                       const bool declare = true);
  void vector_add(InsertionOrderedSet<std::string> &code,
                  const std::string &a,
                  const std::string &b,
                  const std::string &res_name,
                  const int num_components,
                  const bool declare = true);
  void cross_product(InsertionOrderedSet<std::string> &code,
                     const std::string &a,
                     const std::string &b,
                     const std::string &res_name,
                     const int num_components,
                     const bool declare = true);
  void dot_product(InsertionOrderedSet<std::string> &code,
                   const std::string &a,
                   const std::string &b,
                   const std::string &res_name,
                   const int num_components,
                   const bool declare = true);
  void magnitude(InsertionOrderedSet<std::string> &code,
                 const std::string &a,
                 const std::string &res_name,
                 const int num_components,
                 const bool declare = true);
};

class TopologyCode
{
public:
  TopologyCode(const std::string &topo_name, const conduit::Node &domain);
  void vertex_xyz(InsertionOrderedSet<std::string> &code);
  void element_xyz(InsertionOrderedSet<std::string> &code);
  void volume(InsertionOrderedSet<std::string> &code);
  void area(InsertionOrderedSet<std::string> &code);

  // helper functions
  void vertex_idx(InsertionOrderedSet<std::string> &code);
  void element_idx(InsertionOrderedSet<std::string> &code);
  void dxdydz(InsertionOrderedSet<std::string> &code);
  void structured_vertices(InsertionOrderedSet<std::string> &code);
  void unstructured_vertices(InsertionOrderedSet<std::string> &code,
                             const std::string &index_name = "item");
  void vertex_coord(InsertionOrderedSet<std::string> &code,
                    const std::string &coord,
                    const std::string &index_name,
                    const std::string &res_name,
                    const bool declare = true);
  void element_coord(InsertionOrderedSet<std::string> &code,
                     const std::string &coord,
                     const std::string &index_name,
                     const std::string &res_name,
                     const bool declare = true);
  void vertex_xyz(InsertionOrderedSet<std::string> &code,
                  const std::string &index_name,
                  const bool index_array,
                  const std::string &res_name,
                  const bool declare = true);
  void hexahedral_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name);
  void tetrahedral_volume(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name);
  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &p0,
                          const std::string &p1,
                          const std::string &p2,
                          const std::string &p3,
                          const std::string &res_name);
  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name);
  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &p0,
                     const std::string &p1,
                     const std::string &p2,
                     const std::string &res_name);
  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &vertex_locs,
                     const std::string &res_name);
  void polygon_area_vec(InsertionOrderedSet<std::string> &code,
                        const std::string &vertex_locs,
                        const std::string &res_name);
  void polygon_area(InsertionOrderedSet<std::string> &code,
                    const std::string &vertex_locs,
                    const std::string &res_name);
  void polyhedron_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name);
  void hexahedral_surface_area(InsertionOrderedSet<std::string> &code,
                               const std::string &vertex_locs,
                               const std::string &res_name);
  void tetrahedral_surface_area(InsertionOrderedSet<std::string> &code,
                                const std::string &vertex_locs,
                                const std::string &res_name);
  void surface_area(InsertionOrderedSet<std::string> &code);

  std::string topo_name;
  std::string topo_type;
  int num_dims;
  std::string shape;
  int shape_size;

  MathCode math_code;
};

class FieldCode
{
public:
  FieldCode(const std::string &field_name,
            const std::string &topo_name,
            const std::string &association,
            const conduit::Node &domain);
  void gradient(InsertionOrderedSet<std::string> &code);
  void hex_gradient(InsertionOrderedSet<std::string> &code,
                    const std::string &res_name);
  void quad_gradient(InsertionOrderedSet<std::string> &code,
                     const std::string &res_name);
  void field_idx(InsertionOrderedSet<std::string> &code,
                 const std::string &index_name,
                 const std::string &res_name,
                 const std::string &association,
                 const bool declare = true);

  std::string field_name;
  std::string association;
  TopologyCode topo_code;

  MathCode math_code;
};

class Kernel
{
public:
  void fuse_kernel(const Kernel &from);
  std::string generate_output(const std::string &output,
                              bool output_exists) const;
  std::string generate_loop(const std::string &output,
                            const std::string &entries_name) const;

  std::string kernel_body;
  InsertionOrderedSet<std::string> for_body;
  std::string expr;
  // number of components associated with the expression in expr
  // if the expression is a vector expr will just be the name of a single vector
  int num_components;
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
  void execute(conduit::Node &dataset, const std::string &field_name) const;
  std::string generate_kernel(const int dom_idx) const;

  std::unordered_map<std::string, Kernel> kernels;
  // stores entries and argument values for each domain
  conduit::Node dom_info;
  std::string topology;
  std::string association;
  // metadata used to make the . operator work and store various jitable state
  conduit::Node obj;
};

/*
class JitableFunctions
{
  conduit::Node &inputs;
  std::vector<const Kernel *> input_kernels;
  std::vector<const Jitable *> input_jitables;
  std::string &filter_name;
  Kernel &out_kernel;
  Jitable *out_jitable;
  const conduit::Node &dom;
  const int dom_idx;
};
*/
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
