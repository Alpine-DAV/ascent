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
/// file: ascent_web_interface.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_web_interface.hpp"

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
    conduit::relay::about(n);

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

        // support default doc root
        if(m_doc_root == "")
        {
           m_doc_root =  web_client_root_directory();
        }
        // if we aren't using the standard doc root loc, copy
        // the necessary web client files to the requested doc root

        if(m_doc_root != web_client_root_directory())
        {
            copy_directory(web_client_root_directory(),
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

// //-----------------------------------------------------------------------------
// void
// WebInterface::PushImage(PNGEncoder &png)
// {
//     //  Don't do any more work unless we have a valid client connection
//     // (also handles case where stream is not enabled)
//     WebSocket *wsock = Connection();
//
//     if(wsock == NULL)
//     {
//         return;
//     }
//     // create message with image data
//     png.Base64Encode();
//     Node msg;
//     msg["type"] = "image";
//     msg["data"] = "data:image/png;base64," + png.Base64Node().as_string();
//     // sent the message
//     wsock->send(msg);
// }

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
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



