//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_rendering_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_RENDERING_FILTERS
#define ASCENT_RUNTIME_RENDERING_FILTERS

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
/// VTK-H Rendering Filters
///
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class ASCENT_API DefaultRender : public ::flow::Filter
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
class ASCENT_API VTKHBounds: public ::flow::Filter
{
public:
    VTKHBounds();
    virtual ~VTKHBounds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API VTKHUnionBounds: public ::flow::Filter
{
public:
    VTKHUnionBounds();
    virtual ~VTKHUnionBounds();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API CreatePlot : public ::flow::Filter
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
class ASCENT_API AddPlot : public ::flow::Filter
{
public:
    AddPlot();
    virtual ~AddPlot();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();

};

//-----------------------------------------------------------------------------
class ASCENT_API CreateScene : public ::flow::Filter
{
public:
    CreateScene();
    virtual ~CreateScene();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();

};

//-----------------------------------------------------------------------------
class ASCENT_API ExecScene : public ::flow::Filter
{
public:
    ExecScene();

   ~ExecScene();

    virtual void declare_interface(conduit::Node &i);

    virtual void execute();
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
