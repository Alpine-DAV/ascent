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

#ifndef ASCENT_JIT_TOPOLOGY_HPP
#define ASCENT_JIT_TOPOLOGY_HPP

#include "ascent_insertion_ordered_set.hpp"
#include "ascent_jit_math.hpp"
#include "ascent_jit_array.hpp"

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

class TopologyCode
{
public:
  TopologyCode(const std::string &topo_name,
               const conduit::Node &domain,
               const ArrayCode &array_code);

  void pack(conduit::Node &args) const;

  void vertex_xyz(InsertionOrderedSet<std::string> &code) const;
  void element_xyz(InsertionOrderedSet<std::string> &code) const;
  void volume(InsertionOrderedSet<std::string> &code) const;
  void area(InsertionOrderedSet<std::string> &code) const;

  // helper functions
  void vertex_idx(InsertionOrderedSet<std::string> &code) const;

  void element_idx(InsertionOrderedSet<std::string> &code) const;

  void dxdydz(InsertionOrderedSet<std::string> &code) const;

  void structured_vertices(InsertionOrderedSet<std::string> &code) const;

  void structured_vertex_locs(InsertionOrderedSet<std::string> &code) const;

  void unstructured_vertices(InsertionOrderedSet<std::string> &code,
                             const std::string &index_name = "item") const;

  void unstructured_vertex_locs(InsertionOrderedSet<std::string> &code,
                                const std::string &index_name = "item") const;

  void vertex_coord(InsertionOrderedSet<std::string> &code,
                    const std::string &coord,
                    const std::string &index_name,
                    const std::string &res_name,
                    const bool declare = true) const;

  void element_coord(InsertionOrderedSet<std::string> &code,
                     const std::string &coord,
                     const std::string &index_name,
                     const std::string &res_name,
                     const bool declare = true) const;

  void vertex_xyz(InsertionOrderedSet<std::string> &code,
                  const std::string &index_name,
                  const bool index_array,
                  const std::string &res_name,
                  const bool declare = true) const;

  void hexahedral_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name) const;

  void tetrahedral_volume(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name) const;

  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &p0,
                          const std::string &p1,
                          const std::string &p2,
                          const std::string &p3,
                          const std::string &res_name) const;

  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name) const;

  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &p0,
                     const std::string &p1,
                     const std::string &p2,
                     const std::string &res_name) const;

  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &vertex_locs,
                     const std::string &res_name) const;

  void polygon_area_vec(InsertionOrderedSet<std::string> &code,
                        const std::string &vertex_locs,
                        const std::string &res_name) const;

  void polygon_area(InsertionOrderedSet<std::string> &code,
                    const std::string &vertex_locs,
                    const std::string &res_name) const;

  void polyhedron_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name) const;

  void hexahedral_surface_area(InsertionOrderedSet<std::string> &code,
                               const std::string &vertex_locs,
                               const std::string &res_name) const;

  void tetrahedral_surface_area(InsertionOrderedSet<std::string> &code,
                                const std::string &vertex_locs,
                                const std::string &res_name) const;

  void surface_area(InsertionOrderedSet<std::string> &code) const;

  const std::string topo_name;
  const conduit::Node &domain;
  std::string topo_type;
  int num_dims;
  std::string shape;
  int shape_size;

  const ArrayCode &array_code;

  const MathCode math_code;
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
