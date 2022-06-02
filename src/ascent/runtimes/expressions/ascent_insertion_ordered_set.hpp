//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_insertion_ordered_set.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_INSERTION_ORDERED_SET_HPP
#define ASCENT_INSERTION_ORDERED_SET_HPP

#include <vector>
#include <string>
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

// the purpose of this class is to manage duplicates. For example
// inside a derived expression we might have to functions that
// generate code to get the cell center. This class ensures that
// when we add the identical code from the second call, it doesn't
// actually get inserted into the set.
template <typename T>
class InsertionOrderedSet
{
public:
  // Seif: 'unique is meant to bypass the very thing this class is meant
  //        do' if set to false.
  // Matt:  Not sure if its used or needs to exist. Also we can remove
  //        the template since its only ever used with strings
  void insert(const T &item, const bool unique = true);

  void insert(std::initializer_list<T> ilist, const bool unique = true);

  void insert(const InsertionOrderedSet<T> &ios, const bool unique = true);

  T accumulate() const;

  const std::vector<T> & data() const;

private:
  std::unordered_set<T> data_set;
  std::vector<T> insertion_ordered_data;
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
