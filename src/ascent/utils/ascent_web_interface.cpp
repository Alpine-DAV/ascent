//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_web_interface.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_web_interface.hpp"

#include <ascent.hpp>
#include <ascent_config.h>
#include <ascent_file_system.hpp>
#include <ascent_logging.hpp>

// thirdparty includes
#include <lodepng.h>

// conduit includes
#include <conduit_relay.hpp>

using namespace conduit;
using namespace conduit::relay::web;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
std::string
web_client_root_directory()
{
    // check for source dir
    std::string web_root = ASCENT_SOURCE_WEB_CLIENT_ROOT;

    if(conduit::utils::is_directory(web_root))
    {
        return web_root;
    }

    Node n;
    ascent::about(n);

    if(!n.has_child("web_client_root"))
    {
        ASCENT_ERROR("ascent:about result missing 'web_client_root'"
                      << std::endl
                      << n.to_yaml());
    }

    web_root = n["web_client_root"].as_string();

    if(!conduit::utils::is_directory(web_root))
    {
         ASCENT_ERROR("Web client root directory (" << web_root << ") "
                       " is missing");
    }

    return web_root;
}

#ifdef ASCENT_WEBSERVER_ENABLED
//-----------------------------------------------------------------------------
WebInterface::WebInterface()
:m_enabled(false),
 m_ms_poll(100),
 m_ms_timeout(100),
 m_doc_root("")
{}

//-----------------------------------------------------------------------------
WebInterface::~WebInterface()
{
}

//-----------------------------------------------------------------------------
void
WebInterface::SetDocumentRoot(const std::string &path)
{
    m_doc_root = path;
}

//-----------------------------------------------------------------------------
void
WebInterface::SetPoll(int ms_poll)
{
    m_ms_poll = ms_poll;
}

//-----------------------------------------------------------------------------
void
WebInterface::SetTimeout(int ms_timeout)
{
    m_ms_timeout = ms_timeout;
}


void
WebInterface::Enable()
{
    m_enabled = true;
}

//-----------------------------------------------------------------------------
WebSocket *
WebInterface::Connection()
{
    if(!m_enabled)
    {
        return NULL;
    }
    // start our web server if necessary
    if(!m_server.is_running())
    {
        m_server.set_port(8081);

        std::string  default_root =conduit::utils::join_file_path(web_client_root_directory(),
                                                                  "ascent");

        // support default doc root
        if(m_doc_root == "")
        {
           m_doc_root = default_root;
        }
        // if we aren't using the standard doc root loc, copy
        // the necessary web client files to the requested doc root

        if(m_doc_root != default_root)
        {
            copy_directory(default_root,
                           m_doc_root);
        }

        m_server.set_document_root(m_doc_root);
        m_server.serve();
    }

    //  Don't do any more work unless we have a valid client connection
    WebSocket *wsock = m_server.websocket((index_t)m_ms_poll,
                                          (index_t)m_ms_timeout);
    return wsock;
}

//-----------------------------------------------------------------------------
void
WebInterface::PushMessage(const Node &msg)
{
    //  Don't do any more work unless we have a valid client connection
    // (also handles case where stream is not enabled)
    WebSocket *wsock = Connection();

    if(wsock == NULL)
    {
        return;
    }

    // sent the message
    wsock->send(msg);
}

//-----------------------------------------------------------------------------
void
WebInterface::PushRenders(const Node &renders)
{
    //  Don't do any more work unless we have a valid client connection
    // (also handles case where stream is not enabled)
    WebSocket *wsock = Connection();

    if(wsock == NULL)
    {
        return;
    }
    Node msg;

    NodeConstIterator itr = renders.children();

    while(itr.has_next())
    {
        const Node &curr = itr.next();
        EncodeImage(curr.as_string(),
                    msg["renders"].append());

    }


    // sent the message
    wsock->send(msg);
}

//-----------------------------------------------------------------------------
void
WebInterface::EncodeImage(const std::string &png_image_path,
                          conduit::Node &out)
{
    out.reset();

    std::ifstream file(png_image_path.c_str(),
                       std::ios::binary);

    // find out how big the png file is
    file.seekg(0, std::ios::end);
    std::streamsize png_raw_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    // use a node to hold the buffers for raw and base64 encoded png data
    Node png_data;
    png_data["raw"].set(DataType::c_char(png_raw_bytes));
    char *png_raw_ptr = png_data["raw"].value();

    // read in the raw png data
    if(!file.read(png_raw_ptr, png_raw_bytes))
    {
        // ERROR ...
        ASCENT_WARN("ERROR Reading png file " << png_image_path);
    }

    // base64 encode the raw png data
    png_data["encoded"].set(DataType::char8_str(png_raw_bytes*2));

    utils::base64_encode(png_raw_ptr,
                         png_raw_bytes,
                         png_data["encoded"].data_ptr());

    out["data"] = "data:image/png;base64," + png_data["encoded"].as_string();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
#else // else #ifdef ASCENT_WEBSERVER_ENABLED
//-----------------------------------------------------------------------------
// everything web related is stubbed out as a no-op
// the runtime will issue an error message if folks try to enable web streaming
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
WebInterface::WebInterface()
{}

//-----------------------------------------------------------------------------
WebInterface::~WebInterface()
{}

//-----------------------------------------------------------------------------
void
WebInterface::SetDocumentRoot(const std::string &path)
{}

//-----------------------------------------------------------------------------
void
WebInterface::SetPoll(int ms_poll)
{}

//-----------------------------------------------------------------------------
void
WebInterface::SetTimeout(int ms_timeout)
{}

//-----------------------------------------------------------------------------
void
WebInterface::Enable()
{}

//-----------------------------------------------------------------------------
void
WebInterface::PushMessage(const Node &msg)
{}

//-----------------------------------------------------------------------------
void
WebInterface::PushRenders(const Node &renders)
{}

//-----------------------------------------------------------------------------
#endif // end #ifdef ASCENT_WEBSERVER_ENABLED
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



