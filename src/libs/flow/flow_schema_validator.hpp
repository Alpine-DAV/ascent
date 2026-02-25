//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_schema_validator.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_SCHEMA_VALIDATOR_HPP
#define FLOW_SCHEMA_VALIDATOR_HPP

#include <conduit.hpp>

#include <flow_exports.h>
#include <flow_config.h>

//-----------------------------------------------------------------------------
// -- begin flow --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// -- begin flow::schema --
//-----------------------------------------------------------------------------
namespace schema
{

bool FLOW_API validate(const conduit::Node &schema,
                       const conduit::Node &input,
                       conduit::Node &info);


static bool validate_node(const conduit::Node &schema,
                          const conduit::Node &input,
                          conduit::Node &info,
                          const std::string &path);

static bool validate_object(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path);

static bool validate_one_of(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path);

static bool validate_exclusive_children(const conduit::Node &schema,
                                        const conduit::Node &input,
                                        conduit::Node &info,
                                        const std::string &path);

static bool validate_dependencies(const conduit::Node &schema,
                                  const conduit::Node &input,
                                  conduit::Node &info,
                                  const std::string &path);

static bool validate_properties(const conduit::Node &schema,
                                const conduit::Node &input,
                                conduit::Node &info,
                                const std::string &path);

static bool validate_additional_properties(const conduit::Node &schema,
                                           const conduit::Node &input,
                                           conduit::Node &info,
                                           const std::string &path);

static bool validate_required(const conduit::Node &schema,
                              const conduit::Node &input,
                              conduit::Node &info,
                              const std::string &path);

static bool validate_forbid(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path);

static bool check_type(const conduit::Node &input,
                       const conduit::Node &schema,
                       conduit::Node &info,
                       const std::string &path);

static std::string get_type_string(const conduit::Node &schema);

static void add_error(conduit::Node &info, const std::string &msg);

};
//-----------------------------------------------------------------------------
// -- end flow::schema --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------