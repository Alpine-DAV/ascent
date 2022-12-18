//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
