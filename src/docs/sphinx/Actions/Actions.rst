.. ############################################################################
.. # Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Ascent.
.. #
.. # For details, see: http://ascent.readthedocs.io/.
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

Ascent actions can be specified within the integration using Conduit Nodes and can be read in through a file.
Actions files can be defined in both ``json`` or ``yaml``, and if you are human, we recomend using ``yaml``.
Each time Ascent executes a set of actions, it will check for a file in the current working directory called ``ascent_actions.json`` or ``ascent_actions.yaml``.
If found, the current actions specified in code will be replaced with the contents of the json file.
Then default name of the ascent actions file can be specified in the ``ascent_options.json`` or in the
ascent options inside the simulation integration.

Here is an example of an ascent actions json file:

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
              "field": "zonal_noise"
            }
          }
        }
      }
    }
  ]

The equivelent ``yaml`` is

.. code-block:: yaml

  -
    action: "add_scenes"
    scenes:
      scene1:
        plots:
          plt1:
            type: "pseudocolor"
            field: "braid"

A full example of an actions file populated from Ascent's test suite can be found in :ref:`yaml-examples`.


