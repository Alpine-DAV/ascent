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

#if __cplusplus >= 201703L
  #include <optional>
  template <typename T>
  using optional_param = std::optional<T>;
#else
  template <typename T>
  class optional_param
  {
  public:
    optional_param() : m_has_value(false), m_value() {}
    optional_param(const T& value) : m_has_value(true), m_value(value) {}

    operator bool() const { return m_has_value; }

    const T& operator*() const { return m_value; }
    T& operator*() { return m_value; }

  private:
    bool m_has_value;
    T m_value;
  };
#endif

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
                                        optional_param<int> minLength = optional_param<int>(),
                                        optional_param<int> maxLength = optional_param<int>());

conduit::Node ASCENT_API &string_enum_schema(conduit::Node &schema_node, std::vector<std::string> options);

conduit::Node ASCENT_API &bool_schema(conduit::Node &schema_node);

conduit::Node ASCENT_API &number_schema(conduit::Node &schema_node,
                                        bool supports_expressions = false,
                                        optional_param<int> minimum = optional_param<int>(),
                                        optional_param<int> maximum = optional_param<int>(),
                                        optional_param<int> exclusiveMinimum = optional_param<int>(),
                                        optional_param<int> exclusiveMaximum= optional_param<int>());

conduit::Node ASCENT_API &integer_schema(conduit::Node &schema_node,
                                         bool supports_expressions = false,
                                         optional_param<int> minimum = optional_param<int>(),
                                         optional_param<int> maximum = optional_param<int>(),
                                         optional_param<int> exclusiveMinimum = optional_param<int>(),
                                         optional_param<int> exclusiveMaximum= optional_param<int>());

conduit::Node ASCENT_API &vec3_schema(conduit::Node &schema_node,
                                      bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema(conduit::Node &schema_node,
                                      const std::string var1,
                                      const std::string var2,
                                      const std::string var3,
                                      bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema_anyOf(conduit::Node &schema_node,
                                            bool supports_expressions = false);

conduit::Node ASCENT_API &vec3_schema_anyOf(conduit::Node &schema_node,
                                            const std::string var1,
                                            const std::string var2,
                                            const std::string var3,
                                            bool supports_expressions = false);

conduit::Node ASCENT_API &array_schema(conduit::Node &schema_node);

conduit::Node ASCENT_API &array_schema(conduit::Node &schema_node,
                                       const conduit::Node &item_schema);

conduit::Node ASCENT_API &ignore_schema(conduit::Node &schema_node);

bool ASCENT_API check_numeric(const std::string path,
                              const conduit::Node &params,
                              conduit::Node &info,
                              bool required,
                              bool supports_expressions = false);

bool ASCENT_API check_string(const std::string path,
                             const conduit::Node &params,
                             conduit::Node &info,
                             bool required);

bool ASCENT_API check_bool(const std::string path,
                           const conduit::Node &params,
                           conduit::Node &info,
                           bool required);

bool ASCENT_API check_object(const std::string path,
                             const conduit::Node &params,
                             conduit::Node &info,
                             bool required);

bool ASCENT_API check_list(const std::string path,
                           const conduit::Node &params,
                           conduit::Node &info,
                           bool required);

void ASCENT_API path_helper(std::vector<std::string> &paths,
                            const conduit::Node &params);

void ASCENT_API path_helper(std::vector<std::string> &paths,
                            const std::vector<std::string> &ignore,
                            const conduit::Node &params,
                            const std::string path_prefix);

std::string ASCENT_API surprise_check(const std::vector<std::string> &valid_paths,
                                      const conduit::Node &node);

//
// Ignore paths only ignores top level paths, differing lower level
// paths to another surprise check.
//
std::string ASCENT_API surprise_check(const std::vector<std::string> &valid_paths,
                                      const std::vector<std::string> &ignore_paths,
                                      const conduit::Node &node);

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
