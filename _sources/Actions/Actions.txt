.. ############################################################################
.. # Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
.. #
.. # For details, see: http://software.llnl.gov/ascent/.
.. #
.. # Please also read ascent/LICENSE
.. #
.. # Redistribution and use in source and binary forms, with or without
.. # modification, are permitted provided that the following conditions are met:
.. #
.. # * Redistributions of source code must retain the above copyright notice,
.. #   this list of conditions and the disclaimer below.
.. #
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. #
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. #
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. # POSSIBILITY OF SUCH DAMAGE.
.. #
.. ############################################################################

.. _ascent-actions:

Ascent Actions Overview
=======================

Actions are the mechanism that instruct Ascent to perform operations.
The currently supported actions are:

- ``add_scenes``  : adds a list of scenes to create images 
- ``add_extracts``: adds a list of extracts to move data out of Ascent
- ``add_pipelines`` : adds a list of pipelines to transform mesh data
- ``execute`` : executes the data flow network created by the actions
- ``reset`` : resets all actions to an empty state

Ascent actions can be specified within the integration using Conduit Nodes and can be read in through a file.
Each time Ascent executes a set of actions, it will check for a file in the current working directory called ``ascent_actions.json``.
If found, the current actions will be updated with the contents of the json file.
If specific action exists in both the Conduit Node and the file, shared fields will be overridden with the contents of the file.
If additional actions are present in the file, they will be appended to the current list of actions.
The behavior is identical to a Python dictionary update.

For example, if the existing actions in the Conduit Node contain:

.. code-block:: json
  
  [
    
    {
      "action": "add_scenes",
      "scenes": 
      {
        "scene1": 
        {
          "plots": 
          {
            "plt1": 
            {
              "type": "pseudocolor",
              "params": 
              {
                "field": "zonal_noise"
              }
            }
          }
        }
      }
    },
    
    {
      "action": "execute"
    }
  ]


The contents of the file are:

.. code-block:: json

  [
    
    {
      "action": "add_scenes",
      "scenes": 
      {
        "scene1": 
        {
          "plots": 
          {
            "plt2": 
            {
              "type": "pseudocolor",
              "params": 
              {
                "field": "nodal_noise"
              }
            }
          }
        }
      }
    }
  ]

The resulting actions that are executed by Ascent will be:


.. code-block:: json
   
  [
    
    {
      "action": "add_scenes",
      "scenes": 
      {
        "scene1": 
        {
          "plots": 
          {
            "plt1": 
            {
              "type": "pseudocolor",
              "params": 
              {
                "field": "zonal_noise"
              }
            },
            "plt2": 
            {
              "type": "pseudocolor",
              "params": 
              {
                "field": "nodal_noise"
              }
            }
          }
        }
      }
    }
  ]


While the updating feature is convient, we encourage users to be as explicit as possible when creating action files to avoid unexpected behavior.
A full example of an actions file can be found in ``/src/examples/proxies/lulesh2.0.3/ascent_actions.json``.


