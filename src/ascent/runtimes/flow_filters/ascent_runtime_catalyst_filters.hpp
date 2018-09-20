//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
class EnsureVTKH : public ::flow::Filter
{
public:
    EnsureVTKH();
    virtual ~EnsureVTKH();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class EnsureVTKM : public ::flow::Filter
{
public:
    EnsureVTKM();
    virtual ~EnsureVTKM();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class VTKHMarchingCubes : public ::flow::Filter
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
class VTKHSlice : public ::flow::Filter
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
class VTKH3Slice : public ::flow::Filter
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
class VTKHThreshold : public ::flow::Filter
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
class VTKHClip: public ::flow::Filter
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
class VTKHClipWithField : public ::flow::Filter
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
class VTKHIsoVolume : public ::flow::Filter
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
class DefaultRender : public ::flow::Filter
{
public:
    DefaultRender();
    virtual ~DefaultRender();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class VTKHBounds: public ::flow::Filter
{
public:
    VTKHBounds();
    virtual ~VTKHBounds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class VTKHUnionBounds: public ::flow::Filter
{
public:
    VTKHUnionBounds();
    virtual ~VTKHUnionBounds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};



//-----------------------------------------------------------------------------
class VTKHDomainIds: public ::flow::Filter
{
public:
    VTKHDomainIds();
    virtual ~VTKHDomainIds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class VTKHUnionDomainIds: public ::flow::Filter
{
public:
    VTKHUnionDomainIds();
    virtual ~VTKHUnionDomainIds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};


//-----------------------------------------------------------------------------
class DefaultScene: public ::flow::Filter
{
public:
    DefaultScene();
    virtual ~DefaultScene();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

private:
    static int s_image_count;
};

//-----------------------------------------------------------------------------
class CreatePlot : public ::flow::Filter
{
public:
    CreatePlot();
    virtual ~CreatePlot();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

};

//-----------------------------------------------------------------------------
class AddPlot : public ::flow::Filter
{
public:
    AddPlot();
    virtual ~AddPlot();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();

};

//-----------------------------------------------------------------------------
class CreateScene : public ::flow::Filter
{
public:
    CreateScene();
    virtual ~CreateScene();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();

};

//-----------------------------------------------------------------------------
class ExecScene : public ::flow::Filter
{
public:
    ExecScene();

   ~ExecScene();

    virtual void declare_interface(conduit::Node &i);

    virtual void execute();
};
//-----------------------------------------------------------------------------
class VTKHNoOp : public ::flow::Filter
{
public:
    VTKHNoOp();
    virtual ~VTKHNoOp();

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
