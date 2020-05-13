//
// Created by Li, Jixian on 2019-06-04.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H

#include <flow_filter.hpp>
#include <fstream>
#include <sstream>


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
class BFlowPmt : public ::flow::Filter
{
public:
    BFlowPmt() = default;
    virtual ~BFlowPmt() {}

    virtual void   declare_interface(conduit::Node &i) override;
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info) override;
    virtual void   execute() override;
};


//-----------------------------------------------------------------------------
class BFlowVolume : public ::flow::Filter
{
public:
    BFlowVolume() = default;
    virtual ~BFlowVolume() {}

    virtual void   declare_interface(conduit::Node &i) override;
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info) override;
    virtual void   execute() override;
};


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



#endif //ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
