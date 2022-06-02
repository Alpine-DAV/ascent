.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

Ascent Feature Map
====================

These tables provides an inventory of Ascent's features and the programming and data APIs that underpin them.

`Ascent Devil Ray Features <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp#L149>`_

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - Devil Ray Pseudocolor
     - Extract (Rendered Result)
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayPseudocolor <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#L46>`_

   * - Devil Ray 3 Slice
     - Extract (Rendered Result)
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRay3Slice <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#L59>`_

   * - Devil Ray 3 Slice
     - Extract (Rendered Result)
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayVolume <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#72>`_

   * - Devil Ray Project 2D
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayProject2d <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#85>`_

   * - Devil Ray Project Colors 2D
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayProjectColors2d <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#97>`_

   * - Devil Ray Reflect
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayReflect <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#109>`_

   * - Devil Ray Vector Component
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayVectorComponent <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#122>`_



`Ascent VTK-h Features <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp#L105>`_

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - VTK-h Clip
     - Transform
     - VTK-m (OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHClip <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L125>`_

