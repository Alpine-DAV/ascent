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
/// file: ascent_jit_field.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_JIT_FIELD_HPP
#define ASCENT_JIT_FIELD_HPP

#include <memory>
#include <string>
#include "ascent_jit_array.hpp"
#include "ascent_jit_math.hpp"
#include "ascent_jit_topology.hpp"

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

class FieldCode
{
public:
  // if component is -1 use all the field's components
  FieldCode(const std::string &field_name,
            const std::string &association,
            const std::shared_ptr<const TopologyCode> topo_code,
            const ArrayCode &array_code,
            const int num_components,
            const int component);

  void gradient(InsertionOrderedSet<std::string> &code) const;

  void curl(InsertionOrderedSet<std::string> &code) const;

  void recenter(InsertionOrderedSet<std::string> &code,
                const std::string &target_association,
                const std::string &res_name) const;

private:
  // Calculate the element associated gradient of a vertex associated field on
  // a hexahedral mesh
  void hex_gradient(InsertionOrderedSet<std::string> &code,
                    const std::string &res_name) const;

  // Calculate the element associated gradient of a vertex associated field on
  // a quadrilateral mesh
  void quad_gradient(InsertionOrderedSet<std::string> &code,
                     const std::string &res_name) const;

  void element_vertex_values(InsertionOrderedSet<std::string> &code,
                             const std::string &res_name,
                             const int component,
                             const bool declare) const;

  void field_idx(InsertionOrderedSet<std::string> &code,
                 const std::string &index_name,
                 const std::string &association,
                 const std::string &res_name,
                 const bool declare) const;

  void visit_upper(InsertionOrderedSet<std::string> &code,
                   const std::string &index_name,
                   const std::string &if_body,
                   const std::string &else_body,
                   const int dim) const;

  void visit_lower(InsertionOrderedSet<std::string> &code,
                   const std::string &index_name,
                   const std::string &if_body,
                   const std::string &else_body,
                   const int dim) const;

  void visit_current(InsertionOrderedSet<std::string> &code,
                     const std::string &index_name,
                     const std::string &if_body,
                     const std::string &else_body,
                     const int dim) const;

  void visit_vertex_elements(InsertionOrderedSet<std::string> &code,
                             const std::string &index_name,
                             const std::string &if_body,
                             const std::string &else_body,
                             const int dim) const;

  const std::string field_name;
  const std::string association;
  const int num_components;
  const int component;

  const ArrayCode &array_code;

  const std::shared_ptr<const TopologyCode> topo_code;
  const MathCode math_code;
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
