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

bool ASCENT_API check_numeric(const std::string path,
                              const conduit::Node &params,
                              conduit::Node &info,
                              bool required,
                              bool supports_expressions = false);

bool ASCENT_API check_string(const std::string path,
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
// Ignore paths only ignores top level paths, deffering lower level
// paths to another surpise check.
//
std::string ASCENT_API surprise_check(const std::vector<std::string> &valid_paths,
                                      const std::vector<std::string> &ignore_paths,
                                      const conduit::Node &node);

// evalute expression or return value
double ASCENT_API get_float64(const conduit::Node &node, DataObject *dataset);
float ASCENT_API get_float32(const conduit::Node &node, DataObject *dataset);
int ASCENT_API get_int32(const conduit::Node &node, DataObject *dataset);

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
