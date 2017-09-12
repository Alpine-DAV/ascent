//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
WebInterface::WebInterface(int ms_poll,
                           int ms_timeout)
:m_server(NULL),
 m_ms_poll(ms_poll),
 m_ms_timeout(ms_timeout)
{}
  
//-----------------------------------------------------------------------------
WebInterface::~WebInterface()
{
    if(m_server)
    {
        delete m_server;
    }
}


//-----------------------------------------------------------------------------
WebSocket *
WebInterface::Connection()
{
    // start our web server if necessary 
    if(m_server == NULL)
    {
        m_server = new WebServer();
        m_server->set_document_root(ASCENT_WEB_CLIENT_ROOT);
        m_server->set_request_handler(new WebRequestHandler());
        m_server->serve(false);
    }

    //  Don't do any more work unless we have a valid client connection
    WebSocket *wsock = m_server->websocket((index_t)m_ms_poll,
                                           (index_t)m_ms_timeout);
    return wsock;
}

//-----------------------------------------------------------------------------
void
WebInterface::PushMessage(Node &msg)
{
    //  Don't do any more work unless we have a valid client connection
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
WebInterface::PushImage(PNGEncoder &png)
{
    //  Don't do any more work unless we have a valid client connection
    WebSocket *wsock = Connection();

    if(wsock == NULL)
    {
        return;
    }
    // create message with image data
    png.Base64Encode();
    Node msg;
    msg["type"] = "image";
    msg["data"] = "data:image/png;base64," + png.Base64Node().as_string();
    // sent the message
    wsock->send(msg);
}

//-----------------------------------------------------------------------------
void
WebInterface::PushImage(const std::string &png_image_path)
{
    //  Don't do any more work unless we have a valid client connection
    WebSocket *wsock = Connection();

    if(wsock == NULL)
    {
        return;
    }
    
    ASCENT_INFO("png path:" << png_image_path);

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

    // create message with image data
    Node msg;
    msg["type"] = "image";
    msg["data"] = "data:image/png;base64," + png_data["encoded"].as_string();
    // send the message
    wsock->send(msg);
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



