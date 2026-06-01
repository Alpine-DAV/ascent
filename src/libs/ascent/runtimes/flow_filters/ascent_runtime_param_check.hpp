//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_param_check.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_PARAM_CHECK
#define ASCENT_RUNTIME_PARAM_CHECK

#include <conduit.hpp>

#include <map>
#include <string>
#include <vector>

#include <ascent_exports.h>
#include <ascent_data_object.hpp>

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
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

bool ASCENT_API is_valid_expression(const std::string &expr, std::string &err_msg);

void ASCENT_API ascent_register_flow_schema_hooks();

conduit::Node ASCENT_API &string_schema(conduit::Node &schema_node,
                                        const std::size_t minLength = 0,
                                        const std::size_t maxLength = std::numeric_limits<std::size_t>::max());

conduit::Node ASCENT_API &string_enum_schema(conduit::Node &schema_node, const std::vector<std::string> &options);

conduit::Node ASCENT_API &bool_schema(conduit::Node &schema_node);

conduit::Node ASCENT_API &number_schema(conduit::Node &schema_node,
                                        const bool supports_expressions = false,
                                        const int minimum = std::numeric_limits<int>::lowest(),
                                        const int maximum = std::numeric_limits<int>::max(),
                                        const int exclusiveMinimum = std::numeric_limits<int>::lowest(),
                                        const int exclusiveMaximum = std::numeric_limits<int>::max());

conduit::Node ASCENT_API &integer_schema(conduit::Node &schema_node,
                                         const bool supports_expressions = false,
                                         const int minimum = std::numeric_limits<int>::lowest(),
                                         const int maximum = std::numeric_limits<int>::max(),
                                         const int exclusiveMinimum = std::numeric_limits<int>::lowest(),
                                         const int exclusiveMaximum = std::numeric_limits<int>::max());

conduit::Node ASCENT_API &vec3_schema(conduit::Node &schema_node,
                                      const bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema(conduit::Node &schema_node,
                                      const std::string var1,
                                      const std::string var2,
                                      const std::string var3,
                                      const bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema_anyOf(conduit::Node &schema_node,
                                            const bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema_anyOf(conduit::Node &schema_node,
                                            const std::string var1,
                                            const std::string var2,
                                            const std::string var3,
                                            const bool supports_expressions = false);

conduit::Node ASCENT_API &array_schema(conduit::Node &schema_node,
                                       const conduit::Node &item_schema = conduit::Node(),
                                       const std::size_t minItems = 0,
                                       const std::size_t maxItems = std::numeric_limits<std::size_t>::max());

conduit::Node ASCENT_API &ignore_schema(conduit::Node &schema_node);

// evaluate expression or return value
double ASCENT_API get_float64(const conduit::Node &node, DataObject *dataset);
float ASCENT_API  get_float32(const conduit::Node &node, DataObject *dataset);
int ASCENT_API    get_int32(const conduit::Node &node, DataObject *dataset);

//this is for filters that have params that
//can accept either a double value
//or the strings "min" and "max"

enum class ParamVal
{
  Unset,
  Value,
  BoundsMin,
  BoundsMax
};

struct ParamSpec
{
  ParamVal mode = ParamVal::Unset;
  double value = -1.0;
};

//Parse the ParamSpec struct
ParamSpec ASCENT_API assign_param_spec(const conduit::Node &n, DataObject *data_object);

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
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
