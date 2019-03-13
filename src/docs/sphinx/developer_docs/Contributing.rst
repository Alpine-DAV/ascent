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

Ascent Contribution Guide
=========================
The Ascent contribution process is managed through the github repository, and
there are several ways to contribute to the project:

* Issue tracker: questions, issues, and feature requests can be made through the
  project github `issue tracker <https://github.com/Alpine-DAV/ascent/issues>`_
* Email help: help@ascent-dav.org
* Submitting a pull request

Github Code Contributions
-------------------------
The best way to contribute code to improve Ascent is through forking the main repository.
At the top right corner of the Ascent repository is the fork button:

..  figure:: ../images/fork_ascent.png
    :scale: 100 %
    :align: center

After forking, you will have a forked copy of the Ascent repository under your
github account:

..  figure:: ../images/forked_ascent.png
    :scale: 100 %
    :align: center

With a copy of the Ascent repository in hand, you are free to clone your fork
to a local machine and begin development.

What Branch Should I Use?
"""""""""""""""""""""""""
All work on Ascent is done through the ``develop`` branch, which is the default.

What Do I Need To Know About Ascent?
""""""""""""""""""""""""""""""""""""
There are several developer documents that provide a developer guide to add capabilities
to Ascent.

* :ref:`dev_overview`: a developers view of the Ascent
* :ref:`build_env`: how to setup a development environment
* :ref:`vtkh_filter`: developing VTK-m and VTK-h capabilities
* :ref:`flow_filter`: developing flow filters in Ascent

Submitting Pull Requests
""""""""""""""""""""""""
After developing new capability from your forked repository, you can create a pull
request to the main Ascent repository from the forked repo.

..  figure:: ../images/create_pull_request.png
    :scale: 90 %
    :align: center

When submitting the pull request, make sure the pull request is targeting the develop branch.
This should be the default if you created you feature branch from ``develop``.

..  figure:: ../images/submit_pull_request.png
    :scale: 100 %
    :align: center

After submitting the pull request, you can view the pull request from the
main Ascent repository. Additionally, submitting a pull request triggers
Ascent's continuous integration test (CI).

..  figure:: ../images/main_repo_pull_request.png
    :scale: 100 %
    :align: center

If all CI tests pass, the pull request can be reviewed by the Ascent project
members. Only project members can perform the actual merge.

