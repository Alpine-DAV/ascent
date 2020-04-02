//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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
class ASCENT_API VTKHLagrangianInterpolation : public ::flow::Filter
{
public:
    VTKHLagrangianInterpolation();
    virtual ~VTKHLagrangianInterpolation();

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
class ASCENT_API VTKHParticleAdvection : public ::flow::Filter
{
public:
    VTKHParticleAdvection();
    virtual ~VTKHParticleAdvection();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
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
