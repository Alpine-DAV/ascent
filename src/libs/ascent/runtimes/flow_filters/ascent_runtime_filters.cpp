//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_filters.cpp
///
//-----------------------------------------------------------------------------

#include <ascent_runtime_filters.hpp>
#include <ascent_main_runtime.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <flow_workspace.hpp>

#include <ascent_runtime_relay_filters.hpp>
#include <ascent_runtime_htg_filters.hpp>
#include <ascent_runtime_blueprint_filters.hpp>
#include <ascent_runtime_trigger_filters.hpp>
#include <ascent_runtime_query_filters.hpp>

#if defined(ASCENT_VTKM_ENABLED)
    #include <ascent_runtime_vtkh_filters.hpp>
    #include <ascent_runtime_rendering_filters.hpp>
    #include <ascent_runtime_rover_filters.hpp>
#endif

#if defined(ASCENT_DRAY_ENABLED)
    #include <ascent_runtime_dray_filters.hpp>
#endif

#if defined(ASCENT_PYTHON_ENABLED)
    #include <ascent_python_script_filter.hpp>
#endif

#ifdef ASCENT_MPI_ENABLED
    #include <ascent_runtime_hola_filters.hpp>
    #ifdef ASCENT_BABELFLOW_ENABLED
    #include <ascent_runtime_babelflow_filters.hpp>
    #endif
    #ifdef ASCENT_FIDES_ENABLED
    #include <ascent_runtime_adios2_filters.hpp>
    #endif
#endif


#if defined(ASCENT_GENTEN_ENABLED)
   #include <ascent_runtime_genten_filters.hpp>
#endif


using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{


//-----------------------------------------------------------------------------
// init all built in filters
//-----------------------------------------------------------------------------
void
register_builtin()
{
    AscentRuntime::register_filter_type<BlueprintVerify>();
    AscentRuntime::register_filter_type<BlueprintFlatten>("extracts","flatten");
    AscentRuntime::register_filter_type<RelayIOSave>("extracts","relay");
    AscentRuntime::register_filter_type<RelayIOLoad>();
    AscentRuntime::register_filter_type<HTGIOSave>("extracts","htg");

#if defined(ASCENT_GENTEN_ENABLED)
    AscentRuntime::register_filter_type<Learn>("extracts","learn");
#endif

    AscentRuntime::register_filter_type<BasicTrigger>();
    AscentRuntime::register_filter_type<BasicQuery>();
    AscentRuntime::register_filter_type<FilterQuery>("transforms","expression");

    AscentRuntime::register_filter_type<DataBinning>("transforms","binning");
    
    AscentRuntime::register_filter_type<BlueprintPartition>("transforms","partition");

#if defined(ASCENT_VTKM_ENABLED)
    AscentRuntime::register_filter_type<DefaultRender>();

    AscentRuntime::register_filter_type<VTKHBounds>();
    AscentRuntime::register_filter_type<VTKHUnionBounds>();

    // transforms, the current crop expect vtk-h input data
    AscentRuntime::register_filter_type<VTKHClip>("transforms","clip");
    AscentRuntime::register_filter_type<VTKHClipWithField>("transforms","clip_with_field");
    AscentRuntime::register_filter_type<VTKHCleanGrid>("transforms","clean_grid");
    AscentRuntime::register_filter_type<VTKHGhostStripper>("transforms","ghost_stripper");
    AscentRuntime::register_filter_type<VTKHIsoVolume>("transforms","isovolume");
    AscentRuntime::register_filter_type<VTKHLagrangian>("transforms","lagrangian");
    AscentRuntime::register_filter_type<VTKHLog>("transforms","log");
    AscentRuntime::register_filter_type<VTKHLog10>("transforms","log10");
    AscentRuntime::register_filter_type<VTKHLog2>("transforms","log2");
    AscentRuntime::register_filter_type<VTKHMarchingCubes>("transforms","contour");
    AscentRuntime::register_filter_type<VTKHThreshold>("transforms","threshold");
    AscentRuntime::register_filter_type<VTKHSlice>("transforms","slice");
    AscentRuntime::register_filter_type<VTKHAutoSliceLevels>("transforms","auto_slice");
    AscentRuntime::register_filter_type<VTKH3Slice>("transforms","3slice");
    AscentRuntime::register_filter_type<VTKHCompositeVector>("transforms","composite_vector");
    AscentRuntime::register_filter_type<VTKHVectorComponent>("transforms","vector_component");
    AscentRuntime::register_filter_type<VTKHNoOp>("transforms","noop");
    AscentRuntime::register_filter_type<VTKHRecenter>("transforms","recenter");
    AscentRuntime::register_filter_type<VTKHVectorMagnitude>("transforms","vector_magnitude");
    AscentRuntime::register_filter_type<VTKHHistSampling>("transforms","histsampling");
    AscentRuntime::register_filter_type<VTKHQCriterion>("transforms","qcriterion");
    AscentRuntime::register_filter_type<VTKHStats>("extracts","statistics");
    AscentRuntime::register_filter_type<VTKHHistogram>("extracts","histogram");
    AscentRuntime::register_filter_type<VTKHGradient>("transforms","gradient");
    AscentRuntime::register_filter_type<VTKHDivergence>("transforms","divergence");
    AscentRuntime::register_filter_type<VTKHVorticity>("transforms","vorticity");
    AscentRuntime::register_filter_type<VTKHScale>("transforms","scale");
    AscentRuntime::register_filter_type<VTKHProject2d>("transforms","project_2d");
    AscentRuntime::register_filter_type<VTKHTriangulate>("transforms","triangulate");
    AscentRuntime::register_filter_type<VTKHParticleAdvection>("transforms","particle_advection");
    AscentRuntime::register_filter_type<VTKHStreamline>("transforms","streamline");

    AscentRuntime::register_filter_type<RoverXRay>("extracts", "xray");
    AscentRuntime::register_filter_type<RoverVolume>("extracts", "volume");

    AscentRuntime::register_filter_type<AddPlot>();
    AscentRuntime::register_filter_type<CreatePlot>();
    AscentRuntime::register_filter_type<CreateScene>();
    AscentRuntime::register_filter_type<ExecScene>();
#endif

#if defined(ASCENT_DRAY_ENABLED)
    AscentRuntime::register_filter_type<DRayPseudocolor>("extracts", "dray_pseudocolor");
    AscentRuntime::register_filter_type<DRay3Slice>("extracts", "dray_3slice");
    AscentRuntime::register_filter_type<DRayVolume>("extracts", "dray_volume");
    AscentRuntime::register_filter_type<DRayProject2d>("transforms", "dray_project_2d");
    AscentRuntime::register_filter_type<DRayProjectColors2d>("transforms",
                                                             "dray_project_colors_2d");
    AscentRuntime::register_filter_type<DRayReflect>("transforms", "dray_reflect");
    AscentRuntime::register_filter_type<DRayVectorComponent>("transforms", "dray_vector_component");
#endif



#if defined(ASCENT_MPI_ENABLED)
    AscentRuntime::register_filter_type<HolaMPIExtract>("extracts","hola_mpi");

#if defined(ASCENT_BABELFLOW_ENABLED)
    AscentRuntime::register_filter_type<BFlowPmt>("transforms", "bflow_pmt");
    AscentRuntime::register_filter_type<BFlowCompose>("extracts", "bflow_comp");
    AscentRuntime::register_filter_type<BFlowIso>("extracts", "bflow_iso");
#endif

#if defined(ASCENT_ADIOS2_ENABLED)
    AscentRuntime::register_filter_type<ADIOS2>("extracts","adios2");
#endif

#endif

#if defined(ASCENT_PYTHON_ENABLED)
    AscentRuntime::register_filter_type<AscentPythonScript>();
#endif

}



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
