.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _Logging:

Logging Overview
================

Ascent's logging options allow for control over the types of messages and information that are
recorded and displayed. These functionalities are provided to allow for additional information to
be recorded while the tool is being run in order to identify and diagnose any erroneous behavior. 

The ``log`` options refer to the messages that are collected in a designated log stream and output
to a designated log file. The ``echo`` options control the messages and types of information output
to the standard output stream.

Log Levels:
- ``all`` : all messages will be recorded
- ``debug`` : extra verbose output messages useful for diagnosing issues
- ``info`` : normal system operations
- ``warn`` : potential issues that could become a problem
- ``error`` : significant issues that need to be addressed
- ``none`` : no messages will be recorded.

Opening Logs
------------

The ``open_log`` action can be used to start a logging stream. While there are no required keywords,
options to set the output log file name and location using the ``file_pattern`` keyword as well as
the logging threshold level using the ``log_threshold`` are available. The Default ``file_pattern``
is ``ascent_log_output.yaml`` and the default ``log_threshold`` is ``debug``. If using MPI, the
default ``log_threshold`` is ``debug`` for rank 0 and ``warn`` for all other ranks.

.. code-block:: yaml

  -
    action: "open_log"
    file_pattern: "ascent_log_out_{rank:05d}.yaml"
    log_threshold: "debug"

Flushing Logs
-------------

The ``flush_log`` action can be used to flush the current log streams to disk.

.. code-block:: yaml

  -
    action: "flush_log"

Closing Logs
------------

The ``close_log`` action can be used to close the current log stream.

.. code-block:: yaml

  -
    action: "close_log"

Setting the Logging Threshold
-----------------------------

The ``set_log_threshold`` action allows for the adjustment of the level of log messages being
recorded to the current log stream.

.. code-block:: yaml

  -
    action: "set_log_threshold"
    log_threshold: "all"

Setting the Echo Threshold
-----------------------------

The ``set_echo_threshold`` action allows for the adjustment of the level of log messages being
recorded to the standard output stream.

.. code-block:: yaml

  -
    action: "set_echo_threshold"
    echo_threshold: "info"
