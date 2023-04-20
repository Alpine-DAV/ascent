//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_logging.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_LOGGING_HPP
#define ASCENT_LOGGING_HPP

#include <conduit.hpp>
#include <ascent_exports.h>

//-----------------------------------------------------------------------------
//
/// The ASCENT_INFO macro is the primary mechanism used to log basic messages.
///
/// We use CONDUIT_INFO, b/c conduit's logging infrastructure allows use
/// it to be easily rewire messages into a client code's logging mechanism.
///
/// See conduit::utils docs for details.
///
//-----------------------------------------------------------------------------
#define ASCENT_INFO( msg ) std::cout << msg;

//-----------------------------------------------------------------------------
//
/// The ASCENT_WARN macro is the primary mechanism used to capture warnings
/// in ascent.
///
/// We use CONDUIT_WARN, b/c conduit's logging infrastructure allows use
/// it to be easily rewire messages into a client code's logging mechanism.
///
/// See conduit::utils docs for details.
///
//-----------------------------------------------------------------------------
#define ASCENT_WARN( msg ) CONDUIT_WARN( msg );

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

void ASCENT_API handle_error(const std::string &msg,
                             const std::string &file,
                             int line);

//-----------------------------------------------------------------------------
//
/// The ASCENT_ERROR macro is the primary mechanism used to capture errors
/// in ascent.
///
//-----------------------------------------------------------------------------
#define ASCENT_ERROR( msg )                                         \
{                                                                   \
    std::ostringstream ascent_oss_error;                            \
    ascent_oss_error << msg;                                        \
    ::ascent::handle_error( ascent_oss_error.str(),                           \
                  std::string(__FILE__),                            \
                  __LINE__);                                        \
}                                                                   \

} //namespace ascent
#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


