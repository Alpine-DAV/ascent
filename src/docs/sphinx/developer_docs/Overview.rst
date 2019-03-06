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

What Types of Mesh Data Does Ascent Does Ascent Use?
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
where the `h` stands for hybrid-parallel. Inside of VTK-h, we added a distributed-
memory image compositing component and functions that answer global (across all MPI ranks)
questions about data sets such as obtaining the range of a scalar field.

Additionally, VTK-m began as a header only library and VTK-m does not currently build
a library of filters. VTK-h acts as a stand-in for library of VTK-m filters, and VTK-h
maintains the build system that manages CUDA, including GPU device selection, OpenMP, and
Serial compilation. Supporting the range of VTK-m features needed leads to very long
compile times, thus VTK-h insulates Ascent from this additional complexity.

In the future, VTK-m will transition to a fully compiled library, and as distributed-memory
functionality comes online inside VTK-m, we will transition away from VTK-h at some point in
the future.

