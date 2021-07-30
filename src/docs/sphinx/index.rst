.. ############################################################################
.. # Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
.. # Redistribution and use in source and binary forms are permitted, with or without
.. # modification, provided that the following conditions are met:
.. #
.. # * Redistributions of source code must retain the above copyright notice,
.. #   this list of conditions, and the disclaimer below.
.. #
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions, and the disclaimer (as noted below) in the
.. #   documentation and other materials provided with the distribution.
.. #
.. # * Neither the names LLNS and LLNL nor the names of their contributors may
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


========
Ascent
========

A many-core capable flyweight in situ visualization and analysis infrastructure for multi-physics HPC simulations.


.. figure:: images/ascent_examples_small.png

Introduction
============

.. Ascent is a system designed to meet the in-situ visualization and analysis needs of simulation code teams running multi-physics calculations on many-core HPC architectures. It provides rendering runtimes that can leverage many-core CPUs and GPUs to render images of simulation meshes.

Ascent is an easy-to-use flyweight in situ visualization and analysis library for HPC simulations:

- Supports: Making Pictures, Transforming Data, and Capturing Data for use outside of Ascent
- Young effort, yet already includes most common visualization operations
- Provides a simple infrastructure to integrate custom analysis
- Provides C++, C, Python, and Fortran APIs

Ascent's flyweight design targets next-generation HPC platforms:

- Provides efficient distributed-memory (MPI) and many-core (CUDA or OpenMP) execution 
- Demonstrated scaling:  In situ filtering and ray tracing across 16,384 GPUs on LLNL's Sierra Cluster
- Has lower memory requirements than current tools
- Requires less dependencies than current tools (ex: no OpenGL)


Ascent focuses on ease of use and reducing integration burden for simulation code teams:

- Actions are passed to Ascent via YAML files
- Replay capability helps prototype and test actions
- It does not require GUI or system-graphics libraries
- It includes integration examples that demonstrate how to use Ascent inside existing HPC-simulation proxy applications


Getting Started
----------------

To get started building and using Ascent, see the :doc:`Quick Start Guide <QuickStart>`  and the :doc:`Ascent Tutorial <Tutorial>`. For more details about building Ascent see the :doc:`Building documentation <BuildingAscent>`.

Ascent Project Resources
==========================

**Website and Online Documentation**

http://www.ascent-dav.org

**Github Source Repo**

http://github.com/alpine-dav/ascent

**Issue Tracker**

http://github.com/llnl/ascent/issues

**Help Email**

help@ascent-dav.org

Contributors
============

- Cyrus Harrison (LLNL)
- Matt Larsen (LLNL)
- Eric Brugger (LLNL)
- Jim Eliot (AWE)
- Kevin Griffin (LLNL)
- Hank Childs (LBL and UO)
- Utkarsh Ayachit (Kitware, Inc)
- Sudhanshu Sane (UO)
- Seif Ibrahim (LLNL, UCSB)

Ascent Documentation
======================

.. toctree::

   QuickStart
   Ascent
   developer_docs/index.rst
   Tutorial
   Releases
   Publications
   Licenses

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

