//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_anari_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_ANARI_FILTERS
#define ASCENT_RUNTIME_ANARI_FILTERS

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
/// Anari Filters
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// class ASCENT_API AnariIsosurface : public ::flow::Filter
// {
// public:
//     AnariIsosurface();
//     virtual ~AnariIsosurface();
// 
//     virtual void   declare_interface(conduit::Node &i);
//     virtual bool   verify_params(const conduit::Node &params,
//                                  conduit::Node &info);
//     virtual void   execute();
// };

//-----------------------------------------------------------------------------
class ASCENT_API AnariVolume : public ::flow::Filter
{
public:
    AnariVolume();
    virtual ~AnariVolume();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

private:
    struct Impl;
    std::shared_ptr<Impl> pimpl;
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
