.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################


Ascent Overview
=================
The Ascent in situ infrastructure is designed for leading-edge supercomputers,
and has support for both distributed-memory and shared-memory parallelism.
Ascent can take advantage of computing power on both conventional CPU architectures
and on many-core architectures such as NVIDIA and AMD GPUs.
Further, it has a flexible design that supports for integration of new visualization
and analysis routines and libraries. 

The Ascent infrastructure was first presented at
the ISAV 2017 Workshop, held in conjuction with SC 17, in the paper `The ALPINE In Situ
Infrastructure: Ascending from the Ashes of Strawman <https://dl.acm.org/citation.cfm?doid=3144769.3144778>`_.

Ascent is also descried in detail in the book chapter `Ascent: A Flyweight In Situ Library for Exascale Simulations <https://doi.org/10.1007/978-3-030-81627-8_12>`_  In: Childs, H., Bennett, J.C., Garth, C. (eds) In Situ Visualization for Computational Science. Mathematics and Visualization. Springer, Cham. Please use this chapter when :ref:`citing` Ascent.


Requirements
------------
To guide the development of Ascent, we focused on a set of important in situ visualization and analysis requirements extracted from our interactions and experiences with several simulation code teams. Here are Ascent's requirements broken out into three broader categories:

Support a diverse set of simulations on many-core architectures.
  - Support execution on many-core architectures
  - Support  usage  within  a  batch  environment (i.e.,no simulation user involvement once the simulation has begun running).
  - Support the four most common languages used by simulation code teams:  C, C++, Python, and Fortran.
  - Support for multiple data models, including uniform, rectilinear, and unstructured grids.

Provide a streamlined interface to improve usability.
  - Provide  straight  forward  data  ownership  semantics between simulation routines and visualization and analysis routines
  - Provide a low-barrier to entry with respect to developer time for integration.
  - Ease-of-use in terms of directing visualization and analysis actions to occur during runtime.
  - Ease-of-use in terms of consuming visualization results, including delivery mechanisms both for images on a file system and for streaming to a web browser.

Minimize  the  resource  impacts  on  host  simulations.
  - Synchronous in situ processing, meaning that visualization and analysis routines can directly access the memory of a simulation code.
  - Efficient execution times that do not significantly slow down overall simulation time.
  - Minimal memory usage, including zero-copy usage when bringing data from simulation routines to visualization and analysis routines.

System Architecture
-------------------
The Ascent sytem architecture is composed of several components:
  * **Conduit**: `Conduit <http://software.llnl.gov/conduit/>`_  is used to describe and pass in-core mesh data and runtime options from the simulation code to Ascent.
  * **Runtime**: The main Ascent contains runtime implements analysis, rendering, and I/O operations on the mesh data published to Ascent. At a high level, a runtime is responsible for consuming the simulation data that is described using the Conduit Mesh Blueprint and performing a number of actions defined within Conduit Nodes, which create some form of output.
  * **Data Adapters**: Simulation mesh data is described using Conduit's `Mesh Blueprint <http://llnl-conduit.readthedocs.io/en/latest/blueprint.html>`_, which outlines a set of conventions to describe different types of mesh-based scientific data. Each Ascent runtime provides internal Data Adaptors that convert Mesh Blueprint data into a more a more specific data model, such as VTK-m's data model. Ascent will always zero-copy simulation data when possible. To simplify memory ownership semantics, the data provided to Ascent via Conduit Nodes is considered to be owned by the by the simulation.
  * **Embedded Web Server**: Ascent can stream images rendered from a running simulation to a web browser using the Conduit Relay's embedded web-server.

System Diagram
--------------
..  image:: images/AscentSystemDiagram.png
    :width: 85%
    :align: center

