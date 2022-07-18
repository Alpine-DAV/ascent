.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _ascent-actions:

Ascent Actions Overview
=======================

Actions are the mechanism that instruct Ascent to perform operations.
The currently supported actions are:

- ``add_scenes``  : adds a list of scenes to create images
- ``add_extracts``: adds a list of extracts to move data out of Ascent
- ``add_pipelines`` : adds a list of pipelines to transform mesh data
- ``add_queries`` : adds a list of queries that evaluate expressions
- ``add_triggers`` : adds a list of triggers that executes a set of actions based on a condition
- ``save_info`` : saves ascent info result at the end of execution
- ``save_session`` : saves expression session info at the end of execution (see :ref:`ExpressionsSaveSession`)


Ascent actions can be specified within the integration using Conduit Nodes and can be read in through a file.
Actions files can be defined in both ``yaml`` or ``json``, and if you are human, we recommend using ``yaml``.
Each time Ascent executes a set of actions, it will check for a file in the current working directory called ``ascent_actions.yaml`` or ``ascent_actions.json``.
If found, the current actions specified in code will be replaced with the contents of the yaml or json file.
Then default name of the ascent actions file can be specified in the ``ascent_options.yaml`` or in the
ascent options inside the simulation integration.

Here is an example of an ascent actions yaml file:

.. code-block:: yaml

  -
    action: "add_scenes"
    scenes:
      scene1:
        plots:
          plt1:
            type: "pseudocolor"
            field: "braid"

A full example of actions files populated from Ascent's test suite can be found in :ref:`yaml-examples`.


