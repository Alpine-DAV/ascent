//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_BLUEPRINT_TYPE_UTILS
#define ASCENT_BLUEPRINT_TYPE_UTILS

#include <conduit.hpp>
#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>

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
static inline int cell_shape(const std::string &shape_type)
{
  int shape_id = 0;
  if(shape_type == "tri")
  {
      shape_id = 5;
  }
  else if(shape_type == "quad")
  {
      shape_id = 9;
  }
  else if(shape_type == "tet")
  {
      shape_id = 10;
  }
  else if(shape_type == "hex")
  {
      shape_id = 12;
  }
  else if(shape_type == "point")
  {
      shape_id = 1;
  }
  else if(shape_type == "line")
  {
      shape_id = 3;
  }
  else
  {
      ASCENT_ERROR("Unsupported cell type " << shape_type);
  }

  return shape_id;
}

//-----------------------------------------------------------------------------
ASCENT_API bool mcarray_is_float32(const conduit::Node &node);
//-----------------------------------------------------------------------------
ASCENT_API bool mcarray_is_float64(const conduit::Node &node);
//-----------------------------------------------------------------------------
ASCENT_API bool mcarray_is_int32(const conduit::Node &node);
//-----------------------------------------------------------------------------
ASCENT_API bool mcarray_is_int64(const conduit::Node &node);


//-----------------------------------------------------------------------------
ASCENT_API bool field_is_float32(const conduit::Node &field);
//-----------------------------------------------------------------------------
ASCENT_API bool field_is_float64(const conduit::Node &field);
//-----------------------------------------------------------------------------
ASCENT_API bool field_is_int32(const conduit::Node &field);
//-----------------------------------------------------------------------------
ASCENT_API bool field_is_int64(const conduit::Node &field);

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
