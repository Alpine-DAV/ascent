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
/// file: ascent_web_interface.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_WEB_INTERFACE_HPP
#define ASCENT_WEB_INTERFACE_HPP

#include <string>

#include <conduit.hpp>
#include <conduit_relay.hpp>

#include <ascent_png_encoder.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
/// -- Returns path to root directory for web client resources --
/// Source path:     src/ascent/web_clients
/// Installed path:  {install_prefix}/share/ascent/web_clients
//-----------------------------------------------------------------------------
std::string web_client_root_directory();

class WebInterface
{
public:

     WebInterface();
    ~WebInterface();

    // Note: Set methods must be called before first call
    // to Push methods.

    // if set, Ascent's web resources (html, js files, etc) are
    // are copied to and server at the given path
    // if not set, they are served out of web_client_root_directory()

    void                            SetDocumentRoot(const std::string &path);
    void                            SetPoll(int ms_poll);
    void                            SetTimeout(int ms_timeout);

    void                            Enable();

    void                            PushMessage(const conduit::Node &msg);
    void                            PushRenders(const conduit::Node &renders);

private:

    conduit::relay::web::WebSocket *Connection();

    void                            EncodeImage(const std::string &png_file_path,
                                                conduit::Node &out);
    bool                            m_enabled;
    conduit::relay::web::WebServer  m_server;
    int                             m_ms_poll;
    int                             m_ms_timeout;
    std::string                     m_doc_root;

};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


