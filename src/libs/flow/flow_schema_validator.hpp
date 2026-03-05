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

using ExpressionCheckFunc = bool (*)(const std::string &expr, std::string &err_msg);

struct Hooks
{
  ExpressionCheckFunc is_valid_expression = nullptr;
};

bool FLOW_API validate(const conduit::Node &schema,
                       const conduit::Node &input,
                       conduit::Node &info,
                       const Hooks *hooks = nullptr);

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
