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
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHClip <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L125>`_

   * - VTK-h Clip with Field
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHClipWithField <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L138>`_

   * - VTK-h Isovolume
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHIsoVolume <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L151>`_

   * - VTK-h Lagrangian
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHLagrangian <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L164>`_

   * - VTK-h Log
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHLog <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L177>`_

   * - VTK-h Recenter
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHRecenter <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L190>`_

   * - VTK-h Hist Sampling 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHHistSampling <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L203>`_

   * - VTK-h Q Criterion 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHQCriterion <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L216>`_

   * - VTK-h Divergence 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) 
     - VTK-h and VTK-m APIs
     - `VTKHDivergence <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L229>`_

   * - VTK-h Vorticity 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVorticity <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L242>`_

   * - VTK-h Gradient 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHGradient <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L255>`_

   * - VTK-h No Op 
     - Transform
     - None 
     - VTK-h and VTK-m APIs
     - `VTKHNoOp <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L268>`_

   * - VTK-h Vector Component 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVectorComponent <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L281>`_

   * - VTK-h Composite Vector 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHCompositeVector <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L294>`_

   * - VTK-h Statistics 
     - Extract
     - VTK-m (OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHStats <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L307>`_

   * - VTK-h Histogram 
     - Extract
     - VTK-m (OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHHistogram <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L320>`_

   * - VTK-h Project 2D 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHProject2D <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L334>`_

   * - VTK-h Clean Grid 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHCleanGrid <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L348>`_

   * - VTK-h Scale 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHScale <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L361>`_

   * - VTK-h Triangulate 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHTriangulate <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L374>`_

   * - VTK-h Particle Advection 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHParticleAdvection <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L387>`_

   * - VTK-h Streamline 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHStreamline <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L403>`_

   * - VTK-h Contour 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHMarchingCubes <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L47>`_

   * - VTK-h Vector Magnitude 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVectorMagnitude <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L60>`_

   * - VTK-h Slice 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHSlice <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L73>`_

   * - VTK-h 3 Slice 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKH3Slice <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L86>`_

   * - VTK-h Threshold 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHThreshold <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L99>`_

   * - VTK-h Ghost Stripper 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHGhostStripper <https://github.com/Alpine-DAV/ascent/blob/deef65e39f3b2792a40281439c4f614488349c0b/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L112>`_

