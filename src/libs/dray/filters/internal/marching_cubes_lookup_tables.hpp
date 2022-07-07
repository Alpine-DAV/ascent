// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MARCHING_CUBES_LOOKUP_TABLES_HPP
#define DRAY_MARCHING_CUBES_LOOKUP_TABLES_HPP

#include <dray/data_model/elem_attr.hpp>
#include <dray/array.hpp>
#include <dray/vec.hpp>

namespace dray
{

namespace detail
{

#define X -1
const int NO_EDGE = X;

namespace tet
{

const int lookup_size = 7*16 + 16 + 12;
const int ntriangles_offset = 7*16;
const int edges_offset = 7*16 + 16;
const int8 lookup_table[lookup_size] = {
  // Triangle edge definitions
  X, X, X, X, X, X, X,  // Case 0
  0, 3, 2, X, X, X, X,  // Case 1
  0, 1, 4, X, X, X, X,  // Case 2
  1, 4, 2, 2, 4, 3, X,  // Case 3
  1, 2, 5, X, X, X, X,  // Case 4
  0, 3, 5, 0, 5, 1, X,  // Case 5
  0, 2, 5, 0, 5, 4, X,  // Case 6
  5, 4, 3, X, X, X, X,  // Case 7
  3, 4, 5, X, X, X, X,  // Case 8
  4, 5, 0, 5, 2, 0, X,  // Case 9
  1, 5, 0, 5, 3, 0, X,  // Case 10
  5, 2, 1, X, X, X, X,  // Case 11
  3, 4, 2, 2, 4, 1, X,  // Case 12
  4, 1, 0, X, X, X, X,  // Case 13
  2, 3, 0, X, X, X, X,  // Case 14
  X, X, X, X, X, X, X,  // Case 15
  // Number of triangles
  0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0,
  // Edge definitions
  0, 1,  // Edge 0
  1, 2,  // Edge 1
  0, 2,  // Edge 2
  0, 3,  // Edge 3
  1, 3,  // Edge 4
  2, 3   // Edge 5
};

}//namespace tet

// TODO: Define HEX lookup table and functions

#undef X

inline Array<int8>
get_lookup_table(ShapeTet)
{
  using namespace tet;
  Array<int8> retval(lookup_table, lookup_size);
  return retval;
}

DRAY_EXEC const int8 *
get_triangle_edges(ShapeTet, const int8 *table, uint32 flags)
{
  return table + flags*7;
}

DRAY_EXEC int
get_num_triangles(ShapeTet, const int8 *table, uint32 flags)
{
  using namespace tet;
  return static_cast<int>(table[ntriangles_offset + flags]);
}

DRAY_EXEC Vec<int8, 2>
get_edge(ShapeTet, const int8 *table, int edge)
{
  using namespace tet;
  const int8 *offset = table + edges_offset + edge*2;
  Vec<int8, 2> retval;
  retval[0] = offset[0];
  retval[1] = offset[1];
  return retval;
}

}//namespace detail

}//namespace dray

#endif
