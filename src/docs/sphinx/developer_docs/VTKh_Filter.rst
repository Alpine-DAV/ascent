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


VTK-h Filter Anatomy
====================
VTK-h filters can be found in the ``src/vtkh/filters`` directory of the
`VTK-h github repsository <https://github.com/Alpine-DAV/vtk-h>`_.
The VTK-h filter interface is straight-forward:

.. code-block:: c++

  public:
    Filter();
    virtual ~Filter();
    void SetInput(DataSet *input);
    virtual std::string GetName() const = 0;
    DataSet* GetOutput();
    DataSet* Update();

  protected:
    virtual void DoExecute() = 0;
    virtual void PreExecute();
    virtual void PostExecute();

A new VTK-h filter must minimally implement two methods: ``GetName()`` and ``DoExecute``.
A filter's input is a VTK-h data set which is a collection (a std::vector)  of VTK-m data set
with extra meta data like the cycle and domain ids.

Implementing A New Filter
-------------------------
As a convenience, we have created the `NoOp <https://github.com/Alpine-DAV/vtk-h/blob/develop/src/vtkh/filters/NoOp.hpp>`_
filter as staring point. Its recommended that you copy and rename the header and source code
files and use that as a base. The NoOp filter demonstrates how to loop through all the domains
in the input data set, retrieve the underlying VTK-m data set, and where the interesting stuff
goes.

.. code-block:: c++

  void NoOp::DoExecute()
  {
    this->m_output = new DataSet();
    const int num_domains = this->m_input->GetNumberOfDomains();

    for(int i = 0; i < num_domains; ++i)
    {
      vtkm::Id domain_id;
      vtkm::cont::DataSet dom;
      this->m_input->GetDomain(i, dom, domain_id);
      // insert interesting stuff
      m_output->AddDomain(dom, domain_id);
    }
  }

Inside of the source file, you are free to create new and invoke existing VTK-m worklets that will
execute on supported devices. For a more fully functional example, consult the `Marching Cubes <https://github.com/Alpine-DAV/vtk-h/blob/develop/src/vtkh/filters/MarchingCubes.cpp>`_
filter.

Updating the CMakeLists.txt
---------------------------
Once you have copied the header and source file, add the new file names to the CMakeLists file in
the filters directory. The header should be added to the ``vtkh_filters_headers`` list and the
source file to the ``vtkh_filters_sources`` list.

Using MPI Inside VTK-h
----------------------
VTK-h and Ascent both create two separate libraries for MPI and non-MPI (i.e., serial).
In order to maintain the same interface for both version of the library, ``MPI_Comm`` handles
are represented by integers and are converted to the MPI implementations underlying representation
by using the ``MPI_Comm_f2c`` function.

Code containing calls to MPI are protected by the define ``VTKH_PARALLEL`` and calls to MPI API calls
must be guarded inside the code. Here is an example.

.. code-block:: c++

    #ifdef VTKH_PARALLEL
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      int rank;
      MPI_Comm_rank(comm, &rank);
    #endif

.. note::
    ``vtkh::GetMPICommHandle()`` will throw an exception if called outside of a ``VTKH_PARALLEL``
    block.


