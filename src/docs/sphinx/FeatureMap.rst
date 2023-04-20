.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _feature_map:

Ascent Feature Map
====================

These tables provides an inventory of Ascent's features and the programming and data APIs that underpin them.

`Ascent Devil Ray Features <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp#L149>`_

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
     - `DRayPseudocolor <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#L46>`_

   * - Devil Ray 3 Slice
     - Extract (Rendered Result)
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRay3Slice <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#L59>`_

   * - Devil Ray 3 Slice
     - Extract (Rendered Result)
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayVolume <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#72>`_

   * - Devil Ray Project 2D
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayProject2d <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#85>`_

   * - Devil Ray Project Colors 2D
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayProjectColors2d <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#97>`_

   * - Devil Ray Reflect
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayReflect <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#109>`_

   * - Devil Ray Vector Component
     - Transform
     - RAJA + MPI
     - Devil Ray API + MFEM
     - `DRayVectorComponent <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_dray_filters.hpp#122>`_



`Ascent VTK-h Features <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp#L105>`_

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
     - `VTKHClip <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/scent_runtime_vtkh_filters.hpp#L125>`_

   * - VTK-h Clip with Field
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHClipWithField <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L138>`_

   * - VTK-h Isovolume
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHIsoVolume <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L151>`_

   * - VTK-h Lagrangian
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHLagrangian <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L164>`_

   * - VTK-h Log
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHLog <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L177>`_

   * - VTK-h Recenter
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHRecenter <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L190>`_

   * - VTK-h Hist Sampling 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHHistSampling <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L203>`_

   * - VTK-h Q Criterion 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHQCriterion <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L216>`_

   * - VTK-h Divergence 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) 
     - VTK-h and VTK-m APIs
     - `VTKHDivergence <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L229>`_

   * - VTK-h Vorticity 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVorticity <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L242>`_

   * - VTK-h Gradient 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHGradient <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L255>`_

   * - VTK-h No Op 
     - Transform
     - None 
     - VTK-h and VTK-m APIs
     - `VTKHNoOp <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L268>`_

   * - VTK-h Vector Component 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVectorComponent <hhttps://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L281>`_

   * - VTK-h Composite Vector 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHCompositeVector <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L294>`_

   * - VTK-h Statistics 
     - Extract
     - VTK-m (OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHStats <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L307>`_

   * - VTK-h Histogram 
     - Extract
     - VTK-m (OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHHistogram <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L320>`_

   * - VTK-h Project 2D 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHProject2D <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L334>`_

   * - VTK-h Clean Grid 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHCleanGrid <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L348>`_

   * - VTK-h Scale 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHScale <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L361>`_

   * - VTK-h Triangulate 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHTriangulate <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L374>`_

   * - VTK-h Particle Advection 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHParticleAdvection <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L387>`_

   * - VTK-h Streamline 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHStreamline <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L403>`_

   * - VTK-h Contour 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHMarchingCubes <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L47>`_

   * - VTK-h Vector Magnitude 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHVectorMagnitude <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L60>`_

   * - VTK-h Slice 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHSlice <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L73>`_

   * - VTK-h 3 Slice 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKH3Slice <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L86>`_

   * - VTK-h Threshold 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHThreshold <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L99>`_

   * - VTK-h Ghost Stripper 
     - Transform
     - VTK-m (Serial, OpenMP, Cuda, Kokkos)
     - VTK-h and VTK-m APIs
     - `VTKHGhostStripper <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp#L112>`_

   * - VTK-h Mesh Renderer 
     - Extract
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHMeshRenderer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/vtkh/rendering/MeshRenderer.hpp#L9>`_

   * - VTK-h Volume Renderer 
     - Extract
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHVolumeRenderer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/vtkh/rendering/VolumeRenderer.hpp#L15>`_

   * - VTK-h Scalar Renderer 
     - Extract
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHScalarRenderer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/vtkh/rendering/ScalarRenderer.hpp#L16>`_

   * - VTK-h Point Renderer 
     - Extract
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHPointRenderer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/vtkh/rendering/PointRenderer.hpp#L9>`_

   * - VTK-h Line Renderer 
     - Extract
     - VTK-m (Serial, OpenMP, Cuda, Kokkos) + MPI
     - VTK-h and VTK-m APIs
     - `VTKHLineRenderer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/vtkh/rendering/LineRenderer.hpp#L9>`_


`Ascent Expressions  <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/ascent_expression_eval.cpp#L238>`_

.. Expression Lang Primitives 
  .. flow::Workspace::register_filter_type<expressions::Identifier>();
  .. flow::Workspace::register_filter_type<expressions::Double>();
  .. flow::Workspace::register_filter_type<expressions::Integer>();
  .. flow::Workspace::register_filter_type<expressions::String>();
  .. flow::Workspace::register_filter_type<expressions::Boolean>();
  .. flow::Workspace::register_filter_type<expressions::Vector>();
  .. flow::Workspace::register_filter_type<expressions::NullArg>();
  .. flow::Workspace::register_filter_type<expressions::Nan>();

Expression Language Primitives

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - Identifier
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Identifier <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Double
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Double <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Integer
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Integer <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - String
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `String <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Boolean
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Boolean <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Vector
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Vector <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Null
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `NullArg <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Nan
     - Expression Language Primitive
     - C++
     - Conduit Node
     - `Nan <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

.. Expression Lang Operations
  .. flow::Workspace::register_filter_type<expressions::IfExpr>();
  .. flow::Workspace::register_filter_type<expressions::BinaryOp>();
  .. flow::Workspace::register_filter_type<expressions::DotAccess>();
  .. flow::Workspace::register_filter_type<expressions::ArrayAccess>();

Expression Language Operations

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - If Conditional
     - Expression Language Operation
     - C++
     - Conduit Node
     - `IfExpr <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Binary Operation
     - Expression Language Operation
     - C++
     - Conduit Node
     - `BinaryOp <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Dot Access
     - Expression Language Operation
     - C++
     - Conduit Node
     - `DotAccessor <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - Array Access
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArrayAccess <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_


.. History
  .. flow::Workspace::register_filter_type<expressions::History>();
  .. flow::Workspace::register_filter_type<expressions::HistoryRange>();
  .. flow::Workspace::register_filter_type<expressions::ScalarGradient>();
  .. flow::Workspace::register_filter_type<expressions::ArrayGradient>();

History Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `history`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `History <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `history_range`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `HistoryRange <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `scalar_gradient`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ScalarGradient <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `gradient_range`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArrayGradient <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

.. Basic Math
  .. flow::Workspace::register_filter_type<expressions::Abs>();
  .. flow::Workspace::register_filter_type<expressions::Pow>();
  .. flow::Workspace::register_filter_type<expressions::Exp>();
  .. flow::Workspace::register_filter_type<expressions::Log>();
  .. flow::Workspace::register_filter_type<expressions::ScalarMax>();
  .. flow::Workspace::register_filter_type<expressions::ScalarMin>();

Math Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `abs`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Abs <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `exp`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Exp <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `pow`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Pow <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `log`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Log <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `max`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ScalarMax <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `min`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ScalarMin <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

.. Vector Exprs
  .. flow::Workspace::register_filter_type<expressions::Magnitude>();

Vector Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `magnitude`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Magnitude <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_


.. Array Exprs
  .. flow::Workspace::register_filter_type<expressions::ArrayMax>();
  .. flow::Workspace::register_filter_type<expressions::ArrayMin>();
  .. flow::Workspace::register_filter_type<expressions::ArrayAvg>();
  .. flow::Workspace::register_filter_type<expressions::ArraySum>();
  .. flow::Workspace::register_filter_type<expressions::Replace>();


Array Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `array_max`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArrayMax <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `array_min`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArrayMin <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `array_avg`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArraySum <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `array_sum`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `ArraySum <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `replace`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Replace <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `replace`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Replace <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

.. Array Statistics
  .. flow::Workspace::register_filter_type<expressions::Histogram>();
  .. flow::Workspace::register_filter_type<expressions::Entropy>();
  .. flow::Workspace::register_filter_type<expressions::Pdf>();
  .. flow::Workspace::register_filter_type<expressions::Cdf>();
  .. flow::Workspace::register_filter_type<expressions::Quantile>();

Array Statistics Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `histogram`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Histogram <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `entropy`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Entropy <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `pdf`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Pdf <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `cdf`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Cdf <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `quantile`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Quantile <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_


.. Mesh 
  .. flow::Workspace::register_filter_type<expressions::Cycle>();
  .. flow::Workspace::register_filter_type<expressions::Time>();
  .. flow::Workspace::register_filter_type<expressions::Topo>();
  .. flow::Workspace::register_filter_type<expressions::Field>();
  .. flow::Workspace::register_filter_type<expressions::Lineout>();
  .. flow::Workspace::register_filter_type<expressions::Bounds>();
  .. flow::Workspace::register_filter_type<expressions::FieldMax>();
  .. flow::Workspace::register_filter_type<expressions::FieldMin>();
  .. flow::Workspace::register_filter_type<expressions::FieldAvg>();
  .. flow::Workspace::register_filter_type<expressions::FieldSum>();
  .. flow::Workspace::register_filter_type<expressions::FieldNanCount>();
  .. flow::Workspace::register_filter_type<expressions::FieldInfCount>();

Mesh Aware Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `cycle`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Cycle <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `time`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Time <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `topo`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Topo <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Field <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `lineout`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Lineout <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `bounds`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Bounds <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_max`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldMax <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_min`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldMin <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_avg`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldAvg <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_sum`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldSum <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_nan_count`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldNanCount <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `field_inf_count`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `FieldInfCount <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_


.. Binning
  .. flow::Workspace::register_filter_type<expressions::Binning>();
  .. flow::Workspace::register_filter_type<expressions::Axis>();
  .. flow::Workspace::register_filter_type<expressions::Bin>();
  .. flow::Workspace::register_filter_type<expressions::BinByValue>();
  .. flow::Workspace::register_filter_type<expressions::BinByIndex>();
  .. flow::Workspace::register_filter_type<expressions::PointAndAxis>();
  .. flow::Workspace::register_filter_type<expressions::MaxFromPoint>();

Binning Expressions

.. list-table::
   :header-rows: 1

   * - Name
     - Feature Type
     - Programming APIs
     - Data APIs
     - Source Links

   * - `binning` (Mesh Binning)
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Binning <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `axis`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Axis <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `bin`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `Bin <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `bin_by_value`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `BinByValue <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `bin_by_index`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `BinByIndex <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `point_and_axis`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `PointAndAxis <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

   * - `max_from_point`
     - Expression Language Operation
     - C++
     - Conduit Node
     - `MaxFromPoint <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/expressions/ascent_expression_filters.hpp>`_

