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
  using compat_optional = std::optional<T>;
#else
  template <typename T>
  class compat_optional
  {
  public:
    compat_optional() : m_has_value(false), m_value() {}
    compat_optional(const T& value) : m_has_value(true), m_value(value) {}

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

conduit::Node ASCENT_API string_schema(compat_optional<int> minLength = compat_optional<int>(),
                                       compat_optional<int> maxLength = compat_optional<int>());

conduit::Node ASCENT_API string_enum_schema(std::vector<std::string> options);

conduit::Node ASCENT_API bool_schema();

conduit::Node ASCENT_API number_schema(bool supports_expressions = false,
                                       compat_optional<int> minimum = compat_optional<int>(),
                                       compat_optional<int> maximum = compat_optional<int>(),
                                       compat_optional<int> exclusiveMinimum = compat_optional<int>(),
                                       compat_optional<int> exclusiveMaximum= compat_optional<int>());

conduit::Node ASCENT_API integer_schema(bool supports_expressions = false,
                                        compat_optional<int> minimum = compat_optional<int>(),
                                        compat_optional<int> maximum = compat_optional<int>(),
                                        compat_optional<int> exclusiveMinimum = compat_optional<int>(),
                                        compat_optional<int> exclusiveMaximum= compat_optional<int>());

conduit::Node ASCENT_API vec3_schema(bool supports_expressions = false);

conduit::Node ASCENT_API vec3_schema(const std::string var1,
                                     const std::string var2,
                                     const std::string var3,
                                     bool supports_expressions = false);

conduit::Node ASCENT_API vec3_schema_anyOf(bool supports_expressions = false);

conduit::Node ASCENT_API vec3_schema_anyOf(const std::string var1,
                                           const std::string var2,
                                           const std::string var3,
                                           bool supports_expressions = false);

conduit::Node ASCENT_API array_schema();

conduit::Node ASCENT_API array_schema(const conduit::Node &item_schema);

conduit::Node ASCENT_API ignore_schema();

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
