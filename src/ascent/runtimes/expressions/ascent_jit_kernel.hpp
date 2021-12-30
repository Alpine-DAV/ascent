//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_kernel.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_JIT_KERNEL_HPP
#define ASCENT_JIT_KERNEL_HPP

#include "ascent_jit_array.hpp"
#include "ascent_insertion_ordered_set.hpp"

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

class Kernel
{
public:
  void fuse_kernel(const Kernel &from);

  std::string generate_output(const std::string &output,
                              bool output_exists) const;

  std::string generate_loop(const std::string &output,
                            const ArrayCode &array_code,
                            const std::string &entries_name) const;

  InsertionOrderedSet<std::string> functions;
  InsertionOrderedSet<std::string> kernel_body;
  InsertionOrderedSet<std::string> for_body;
  std::string expr;
  // number of components associated with the expression in expr
  // if the expression is a vector expr will just be the name of a single vector
  int num_components;
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
