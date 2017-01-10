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
.. # For details, see: http://software.llnl.gov/strawman/.
.. #
.. # Please also read strawman/LICENSE
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


========
Strawman
========

A many-core capable lightweight in situ visualization and analysis infrastructure for multi-physics HPC simulations.

Introduction
============

Strawman is a system designed to explore the in situ visualization and analysis needs of simulation code teams running multi-physics calculations on many-core HPC architectures. It provides rendering pipelines that can leverage both many-core CPUs and GPUs to render images of simulation meshes. 

Strawman focuses on ease of use and reduced integration burden for simulation code teams:

- It does not require any GUI or system graphics libraries.
- It includes integration examples which demonstrate how to use Strawman inside of three different HPC simulation proxy applications.
- It provides a built-in web server that supports streaming rendered images directly to a web browser.

Strawman Project Resources
==========================

**Online Documentation**

http://software.llnl.gov/strawman

**Githib Source Repo**

http://github.com/llnl/strawman

**Issue Tracker**

http://github.com/llnl/strawman/issues

Contributors
============

- Cyrus Harrison (LLNL)
- Matt Larsen (LLNL)
- Eric Brugger (LLNL)
- Jim Eliot (AWE)
- Kevin Griffin (LLNL)
- Hank Childs (LBL and UO)

Strawman Documentation
======================

.. toctree::

   Strawman 
   Releases
   Publications
   Licenses

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

