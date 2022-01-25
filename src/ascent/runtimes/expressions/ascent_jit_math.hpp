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

#ifndef ASCENT_JIT_MATH_HPP
#define ASCENT_JIT_MATH_HPP

#include "ascent_insertion_ordered_set.hpp"
#include <string>

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

class MathCode
{
public:
  void determinant_2x2(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const bool declare = true) const;

  void determinant_3x3(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &c,
                       const std::string &res_name,
                       const bool declare = true) const;

  void vector_subtract(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const int num_components,
                       const bool declare = true) const;

  void vector_add(InsertionOrderedSet<std::string> &code,
                  const std::string &a,
                  const std::string &b,
                  const std::string &res_name,
                  const int num_components,
                  const bool declare = true) const;

  void cross_product(InsertionOrderedSet<std::string> &code,
                     const std::string &a,
                     const std::string &b,
                     const std::string &res_name,
                     const int num_components,
                     const bool declare = true) const;

  void dot_product(InsertionOrderedSet<std::string> &code,
                   const std::string &a,
                   const std::string &b,
                   const std::string &res_name,
                   const int num_components,
                   const bool declare = true) const;

  void magnitude(InsertionOrderedSet<std::string> &code,
                 const std::string &a,
                 const std::string &res_name,
                 const int num_components,
                 const bool declare = true) const;

  void array_avg(InsertionOrderedSet<std::string> &code,
                 const int length,
                 const std::string &array_name,
                 const std::string &res_name,
                 const bool declare) const;

  void component_avg(InsertionOrderedSet<std::string> &code,
                     const int length,
                     const std::string &array_name,
                     const std::string &coord,
                     const std::string &res_name,
                     const bool declare) const;
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
