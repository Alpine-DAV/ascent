//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
#include <ascent_config.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
/// -- Returns path to root directory for web client resources --
/// Source path:     src/libs/ascent/web_clients
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
#ifdef ASCENT_WEBSERVER_ENABLED
    conduit::relay::web::WebSocket *Connection();

    void                            EncodeImage(const std::string &png_file_path,
                                                conduit::Node &out);
    bool                            m_enabled;
    conduit::relay::web::WebServer  m_server;
    int                             m_ms_poll;
    int                             m_ms_timeout;
    std::string                     m_doc_root;
#endif
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


