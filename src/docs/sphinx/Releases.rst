.. ############################################################################
.. # Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
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

Releases
========

Ascent and its dependencies are under rapid development.
Because of this we recommend using our develop branch, which we aim 
to keep buildable via continuous integration testing. See our 
:doc:`Quick Start Guide <QuickStart>` for info on how to build Ascent and 
its dependencies.



Source distributions for Ascent are hosted on github:

https://github.com/Alpine-DAV/ascent/releases

v0.7.1
-------

* `v0.7.1 Source Tarball <https://github.com/Alpine-DAV/ascent/releases/download/v0.7.1/ascent-v0.7.1-src-with-blt.tar.gz>`_

Highlights
+++++++++++++

(Extracted from Ascent's :download:`Changelog <../../../CHANGELOG.md>`)

Preferred dependency versions for ascent@0.7.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* conduit@0.7.2
* dray@0.1.6
* vtk-h@0.7.1
* vtk-m@1.5.5


Added
~~~~~~~~~
* Added Data Binning examples to the Ascent Intro Tutorial

Fixed
~~~~~~~~~
* Fixed an issue with the Data Binning bin calculation logic

Changed
~~~~~~~~~
* Updated Ascent to use new conduit, dray, and vtk-h versions



v0.7.0
-------

* `v0.7.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases/download/v0.7.0/ascent-v0.7.0-src-with-blt.tar.gz>`_

Highlights
+++++++++++++

(Extracted from Ascent's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* Added partial failure tolerance (i.e., if there are multiple plots the failure of one doesn't prevent the others from rendering)
* Added the ability to use expressions as parameters to filters, e.g., ``iso contour value = "(max(field('density')) - min(field('density)) / 2")``
* Added orthogonal projections for scalar images (projecting onto a 2d plane)
* Added a `triangulate` transform
* Added option to build Ascent with only Devil Ray support

Fixed
~~~~~~~~~

* Fixed a MPI hang if actions files (yaml or json) fail to parse
* Fixed several minor issues with saving and reading Mesh Blueprint file sets
* Fixed a field association bug with Data Binning
* Fixed a 2D AMR mesh rendering issue

Changed
~~~~~~~~~

* To better support installs that are relocated on the file system, Cinema database file resources are now compiled into the Ascent library.
* Updated to use Babelflow (1.0.1) and Parallel Merge Tree (1.0.2).



v0.6.0
-------

* `v0.6.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases/download/v0.6.0/ascent-v0.6.0-src-with-blt.tar.gz>`_

Highlights
+++++++++++++

(Extracted from Ascent's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* Added support for Devil Ray (high-order) ray tracer
* Added vector operations
  * composite vector (create vector from three scalars)
  * vector component (extract scalar component)
* Allow no refinement for high-order meshes
* Added support for multiple topologies (e.g., volume and particles in the same mesh)
* Added support for AMR Nesting relationships (Blueprint Nestsets)
* Added optional ``num_files`` parameter to the Relay Extract. See the [Relay Extract Docs](https://ascent.readthedocs.io/en/latest/Actions/Extracts.html#relay) for more details.
* Added an AscentViewer Widget for Jupyter
* Added new CUDA device link logic to help bottle CUDA dependencies for downstream use
* Added support for `exa` prefix style filters


Changed
~~~~~~~~~
* Modified Cinema output so it can be viewed without a webserver
* Removed default behavior of publishing individual vector components when vectors were three separate arrays. This can be achieved by using the vector component filter
* Changed Docker Images to leverage Jupyter lab
* Tutorial updates
* Rendering improvements


v0.5.1
-------

* `v0.5.1 Source Tarball <https://github.com/Alpine-DAV/ascent/releases/download/v0.5.1/ascent-v0.5.1-src-with-blt.tar.gz>`_

Highlights
+++++++++++++

(Extracted from Ascent's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* Added support to render multiple topologies in the same scene.
* Added a Data Object construct to the main Ascent runtime to easily manage transformations between in-memory mesh representations. 

Fixed
~~~~~~~~~
* Issue where cycle was not properly propagated when converting mfem data.
* Cinema issue where zoom was applied additively each cycle to oblivion.
* Cinema issue where cameras were not following the center of the data set.

v0.5.0
-------

* `v0.5.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases/download/v0.5.0/ascent-v0.5.0-src-with-blt.tar.gz>`_

Highlights
+++++++++++++

(Extracted from Ascent's :download:`Changelog <../../../CHANGELOG.md>`)

Added
~~~~~~~~~

* Added new :ref:`Tutorial Content <tutorial_intro>` including C++, Python, and Python-based Jupyter Notebook examples.
* Added docs for :ref:`queries` and :ref:`triggers`
* Added a Jupyter Extract that provides interactive Python Notebook access to published mesh data. See the related :ref:`Cloverleaf Demo <cloverleaf_demo_jupyter_extract>`.
* Deprecated the `execute` and `reset` actions. `ascent.execute(actions)` now implicitly resets and execute the Ascent actions. To maintain a degree of backwards compatibility, using `execute` and `reset` are still passable to `ascent.execute(actions)`. Internally, the internal data flow network will only be rebuilt when the current actions differ from the previously executed actions. Note: this only occurs when the Ascent runtime object is persistent between calls to `ascent.execute(actions)`.
* Added support for YAML `ascent_actions` and `ascent_options` files. YAML files are much easier for humans to compose
* Add a relative offset option to the Slice filter.

Changed
~~~~~~~~~

* Several improvements to Ascent's Expression infrastructure.
* Updated our uberenv-based to use a 2019/11 version of spack develop.
* Improved Python error handling and propagation.
* Updated Docker example to build with Jupyter Notebook support.
* Updated to VTK-m 1.5.0 and associated VTK-h.
* Imposed necessary static build constraints for cuda support.

Fixed
~~~~~~~~~

* Several minor bug fixes 


v0.4.0
-------

The fourth release of Ascent.

* `v0.4.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases>`_
  
v0.3.0
-------

The third release of Ascent.

* `v0.3.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases>`_

v0.2.0
-------

The second release of Ascent.

* `v0.2.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases>`_

v0.1.0
-------

The initial release of Ascent.

* `v0.1.0 Source Tarball <https://github.com/Alpine-DAV/ascent/releases>`_




