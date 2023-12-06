//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_blueprint_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_BLUEPRINT_FILTERS
#define ASCENT_RUNTIME_BLUEPRINT_FILTERS

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
/// Filters Related to Blueprint
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ASCENT_API BlueprintVerify : public ::flow::Filter
{
public:
    BlueprintVerify();
   ~BlueprintVerify();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
// In-memory conduit extract, published to registry
//-----------------------------------------------------------------------------
class ASCENT_API ConduitExtract: public ::flow::Filter
{
public:
    ConduitExtract();
   ~ConduitExtract();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};


//-----------------------------------------------------------------------------
class ASCENT_API BlueprintPartition : public ::flow::Filter
{
public:
    BlueprintPartition();
   ~BlueprintPartition();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API DataBinning : public ::flow::Filter
{
public:
    DataBinning();
   ~DataBinning();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
class ASCENT_API AddFields : public ::flow::Filter
{
public:
    AddFields();
   ~AddFields();

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
