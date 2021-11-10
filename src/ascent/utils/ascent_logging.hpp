//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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


