//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_insertion_ordered_set.cpp
///
//-----------------------------------------------------------------------------

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

template<typename T>
void
InsertionOrderedSet<T>::insert(const T &item, const bool unique)
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

template<typename T>
void
InsertionOrderedSet<T>::insert(std::initializer_list<T> ilist, const bool unique)
{
  for(const auto &item : ilist)
  {
    insert(item, unique);
  }
}

template<typename T>
void
InsertionOrderedSet<T>::insert(const InsertionOrderedSet<T> &ios, const bool unique)
{
  for(const auto &item : ios.data())
  {
    insert(item, unique);
  }
}

template<typename T>
T
InsertionOrderedSet<T>::accumulate() const
{
  T res;
  for(const auto &item : insertion_ordered_data)
  {
    res += item;
  }
  return res;
}

template<typename T>
const std::vector<T> &
InsertionOrderedSet<T>::data() const
{
  return insertion_ordered_data;
}

template class InsertionOrderedSet<std::string>;

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
