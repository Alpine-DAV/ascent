//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_vtkh_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_VTKH_FILTERS
#define ASCENT_RUNTIME_VTKH_FILTERS

#include <ascent.hpp>

#include <flow_filter.hpp>


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
///
/// VTK-H Filters
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ASCENT_API VTKHMarchingCubes : public ::flow::Filter
{
public:
    VTKHMarchingCubes();
    virtual ~VTKHMarchingCubes();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHVectorMagnitude : public ::flow::Filter
{
public:
    VTKHVectorMagnitude();
    virtual ~VTKHVectorMagnitude();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHSlice : public ::flow::Filter
{
public:
    VTKHSlice();
    virtual ~VTKHSlice();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKH3Slice : public ::flow::Filter
{
public:
    VTKH3Slice();
    virtual ~VTKH3Slice();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHThreshold : public ::flow::Filter
{
public:
    VTKHThreshold();
    virtual ~VTKHThreshold();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHGhostStripper: public ::flow::Filter
{
public:
    VTKHGhostStripper();
    virtual ~VTKHGhostStripper();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHClip: public ::flow::Filter
{
public:
    VTKHClip();
    virtual ~VTKHClip();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHClipWithField : public ::flow::Filter
{
public:
    VTKHClipWithField();
    virtual ~VTKHClipWithField();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHIsoVolume : public ::flow::Filter
{
public:
    VTKHIsoVolume();
    virtual ~VTKHIsoVolume();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHLagrangian : public ::flow::Filter
{
public:
    VTKHLagrangian();
    virtual ~VTKHLagrangian();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHLog: public ::flow::Filter
{
public:
    VTKHLog();
    virtual ~VTKHLog();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHLog10: public ::flow::Filter
{
public:
    VTKHLog10();
    virtual ~VTKHLog10();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHLog2: public ::flow::Filter
{
public:
    VTKHLog2();
    virtual ~VTKHLog2();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHRecenter: public ::flow::Filter
{
public:
    VTKHRecenter();
    virtual ~VTKHRecenter();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHHistSampling : public ::flow::Filter
{
public:
    VTKHHistSampling();
    virtual ~VTKHHistSampling();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHQCriterion: public ::flow::Filter
{
public:
    VTKHQCriterion();
    virtual ~VTKHQCriterion();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHDivergence: public ::flow::Filter
{
public:
    VTKHDivergence();
    virtual ~VTKHDivergence();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHVorticity: public ::flow::Filter
{
public:
    VTKHVorticity();
    virtual ~VTKHVorticity();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHGradient : public ::flow::Filter
{
public:
    VTKHGradient();
    virtual ~VTKHGradient();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHNoOp : public ::flow::Filter
{
public:
    VTKHNoOp();
    virtual ~VTKHNoOp();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHVectorComponent : public ::flow::Filter
{
public:
    VTKHVectorComponent();
    virtual ~VTKHVectorComponent();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHCompositeVector : public ::flow::Filter
{
public:
    VTKHCompositeVector();
    virtual ~VTKHCompositeVector();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHStats : public ::flow::Filter
{
public:
    VTKHStats();
    virtual ~VTKHStats();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHHistogram : public ::flow::Filter
{
public:
    VTKHHistogram();
    virtual ~VTKHHistogram();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};


//-----------------------------------------------------------------------------
class ASCENT_API VTKHProject2d : public ::flow::Filter
{
public:
    VTKHProject2d();
    virtual ~VTKHProject2d();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};


//-----------------------------------------------------------------------------
class ASCENT_API VTKHCleanGrid : public ::flow::Filter
{
public:
    VTKHCleanGrid();
    virtual ~VTKHCleanGrid();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHScale : public ::flow::Filter
{
public:
    VTKHScale();
    virtual ~VTKHScale();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHTriangulate : public ::flow::Filter
{
public:
    VTKHTriangulate();
    virtual ~VTKHTriangulate();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHParticleAdvection : public ::flow::Filter
{
public:
    VTKHParticleAdvection();
    virtual ~VTKHParticleAdvection();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

protected:
    bool record_trajectories = false;
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHStreamline : public VTKHParticleAdvection
{
public:
  VTKHStreamline();
  virtual ~VTKHStreamline();
  virtual void   declare_interface(conduit::Node &i);
};

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




#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
