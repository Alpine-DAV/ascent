//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
