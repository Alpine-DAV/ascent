//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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


function strawman_websocket_client()
{
    var wsproto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    connection = new WebSocket(wsproto + '//' + window.location.host + '/websocket');
    
    connection.onmessage = function (json) 
    {
        $("#status_label").html('<span class="label label-success">Connected</span>')
        var msg;
        try
        {
            msg=JSON.parse(json.data);
        }
        catch(e)
        {
            console.log(e);
        }

        if(msg.type == "image")
        {
            $("#render_display").html("<img src='" + msg.data + "' width=500 height=500/>");
            $("#render_display").show();

            $("#connection_info").html('<span class="label label-success">Connected</span>');
        }
        else if(msg.type == "status")
        {
            $("#status").html("<b>[Simulation State]</b><br><b>time:</b> " + msg.data.time.toFixed(6)  + " <br> <b>cycle:</b> " + msg.data.cycle);

            if(msg.data.hasOwnProperty('info'))
            {
                $("#info").html("<h2>" + msg.data.info + "</h2>");
            }
            else
            {
                $("#info").html("");
            }

        }
    }
      
    connection.onerror = function (error)
    {
        console.log('WebSocket error');
        connection.close();
        $("#connection_info").html('<span class="label label-warning">Not Connected</span>');
    }
    
    connection.onclose = function (error)
    {
        console.log('WebSocket closed');
        console.log(error)
        connection.close();
        $("#connection_info").html('<span class="label label-warning">Not Connected</span>');
        // this allows infinite reconnect ...
        strawman_websocket_client();
    }

    
}

strawman_websocket_client();




