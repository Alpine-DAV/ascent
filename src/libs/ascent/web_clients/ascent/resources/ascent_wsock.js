//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/*
highlight_json adapted from:
https://stackoverflow.com/questions/4810841/how-can-i-pretty-print-json-using-javascript
*/
function highlight_json(json)
{
    if (typeof json != 'string')
    {
         json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    json = json.replace(/\n/g, "<br>").replace(/[ ]/g, "&nbsp;");
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function ascent_websocket_client()
{
    var num_msgs = 0;
    var wsproto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    connection = new WebSocket(wsproto + '//' + window.location.host + '/websocket');

    connection.onmessage = function (json)
    {
        $("#connection_info").html('<span class="badge badge-pill badge-primary">Connected</span>')
        var msg;
        try
        {
            msg=JSON.parse(json.data);
            //console.log(msg);
        }
        catch(e)
        {
            console.log(e);
        }

        num_msgs+=1;
        $("#status").html("# msgs: " +  num_msgs.toString());

        if(msg.info)
        {
            $("#info").html(highlight_json(msg));
        }

        if(msg.renders)
        {
            // loop over renders
            var num_renders = msg.renders.length;
            var res = "";
            for (var i = 0; i < num_renders; i++)
            {
                res += "<img src='" + msg.renders[i].data + "' width=500 height=500/>"
            }

            $("#render_display").html(res);
            $("#render_display").show();
        }
    }

    connection.onerror = function (error)
    {
        console.log('WebSocket error');
        connection.close();
        $("#connection_info").html('<span class="badge badge-pill badge-secondary">Not Connected</span>')
    }

    connection.onclose = function (error)
    {
        console.log('WebSocket closed');
        console.log(error)
        connection.close();
        $("#connection_info").html('<span class="badge badge-pill badge-secondary">Not Connected</span>')
        // this allows infinite reconnect ...
        ascent_websocket_client();
    }


}

ascent_websocket_client();




