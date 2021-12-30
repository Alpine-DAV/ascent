.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################


.. _dev_overview:

Ascent Developement Overview
============================
Ascent's architecture is divided into two main components:

  * Flow: a simple and flexible data flow network
  * Runtime: code that assembles and runs data flow networks to process data

Flow
----
Ascent uses a simple data flow library named Flow to efficiently
compose and execute filters. Ascent's Flow library is a C++
evolution of the Python data flow network infrastructure used in
`Efficient Dynamic Derived Field Generation on Many-Core Architectures Using Python <https://ieeexplore.ieee.org/document/6495864>`_.
Flow supports declaration and execution of directed acyclic
graphs (DAGs) of filters. Filters declare a minimal interface, which
includes the number of expected inputs and outputs, and a set of default
parameters. Flow uses a topological sort to ensure proper filter
execution order, tracks all intermediate results, and provides
basic memory management capabilities.

There are three main components to Flow:

  * Registry: manages the lifetime of intermediate filter results
  * Graph: contains the filter graph and manages the adding of filters
  * Workspace: contains both the registry and filter graph

Flow filters are the basic unit of execution inside of Ascent, and all functionality
is implemented as a Flow filter.

Ascent Runtime
--------------
The Ascent runtime accepts user actions described in Conduit nodes and
uses that information to assemble a data flow network. Outside the
Ascent runtime, all functionality is wrapped and executed through Flow
filters, and the Ascent runtime logically divides flow filters into
two categories:

  * Transform: a Flow filter that transforms data
  * Extract: a sink Flow filter that consumes simulation data or the results of pipelines

Most developers will create either a transform or an extract. Flow filters are
registered with the Ascent runtime by declaring the type of the filter
(extract or transform), and the API name that users can specify in the Ascent actions.

.. note::
    Flow filters can also be registered with the Ascent runtime by applications outside of Ascent.

What Types of Mesh Data Does Ascent Use?
----------------------------------------------------
Ascent supports several different data types, and has adapters for converting between them.

  * Conduit Mesh Blueprint: used to publish data to ascent
  * VTK-h: a simple collection of VTK-m data sets
  * MFEM: high-order finite element meshes

Implementers of Flow filters must check the input data type and apply the
appropriate conversions if the data does not match what is required.

Mesh Types
""""""""""
The following mesh types are supported in Ascent:

  * Uniform
  * Rectilinear
  * Curvilinear
  * Explicit
  * High-Order (Blueprint and MFEM)

High-order mesh can be converted to low-order through a filter. By default,
all mesh data for transforms is already converted to low-order meshes.

Domain Overloading
""""""""""""""""""
Ascent supports arbitrary domain overloading, so all filters should support
multiple domains per rank. Additionally, there is no guarantee that a rank will have
any data at all, especially after a series of transformations.

VTK-m
-----
Ascent's ability to perform visualization operations on exascale architectures
is underpinned by VTK-m. Currently, pipelines in Ascent are constructed with various
VTK-m filters wrapped by VTK-h and then by a flow filter. Although strongly encouraged,
Ascent does not need to be compiled with VTK-m support.

VTK-h
-----
At the beginning of Ascent development, there was no support for MPI inside of
VTK-m. To augment VTK-m with distributed-memory capabilities, we created VTK-h,
where the `h` stands for hybrid-parallel. Inside of VTK-h, we added a distributed-memory
image compositing component and functions that answer global (across all MPI ranks)
questions about data sets such as obtaining the range of a scalar field.

Additionally, VTK-m began as a header only library and VTK-m does not currently build
a library of filters. VTK-h acts as a stand-in for a library of VTK-m filters, and VTK-h
maintains the build system that manages CUDA, including GPU device selection, OpenMP, and
Serial compilation. Supporting the range of VTK-m features needed leads to very long
compile times, thus VTK-h insulates Ascent from this additional complexity.

In the future, VTK-m will transition to a fully compiled library, and as distributed-memory
functionality comes online inside VTK-m, we will transition away from VTK-h at some point in
the future.

