//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ascent_blueprint_type_utils.hpp"

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

//-----------------------------------------------------------------------------
bool mcarray_is_float32(const conduit::Node &node)
{
  const int children = node.number_of_children();
  if(children == 0)
  {
    return node.dtype().is_float32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return node.child(0).dtype().is_float32();
  }
}

//-----------------------------------------------------------------------------
bool mcarray_is_float64(const conduit::Node &node)
{
  const int children = node.number_of_children();
  if(children == 0)
  {
    return node.dtype().is_float64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return node.child(0).dtype().is_float64();
  }
}

//-----------------------------------------------------------------------------
bool mcarray_is_int32(const conduit::Node &node)
{
  const int children = node.number_of_children();
  if(children == 0)
  {
    return node.dtype().is_int32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return node.child(0).dtype().is_int32();
  }
}

//-----------------------------------------------------------------------------
bool mcarray_is_int64(const conduit::Node &node)
{
  const int children = node.number_of_children();
  if(children == 0)
  {
    return node.dtype().is_int64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return node.child(0).dtype().is_int64();
  }
}

//-----------------------------------------------------------------------------
bool field_is_float32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float32();
  }
}

//-----------------------------------------------------------------------------
bool field_is_float64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float64();
  }
}

//-----------------------------------------------------------------------------
bool field_is_int32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int32();
  }
}

//-----------------------------------------------------------------------------
bool field_is_int64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int64();
  }
}

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

