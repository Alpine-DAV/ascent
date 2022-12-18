//
// Created by Sergei Shudler on 2020-06-09.
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
class BFlowCompose : public ::flow::Filter
{
public:
    enum CompositingType { REDUCE = 0, BINSWAP = 1, RADIX_K = 2 };
    
    BFlowCompose() = default;
    virtual ~BFlowCompose() {}

    virtual void   declare_interface(conduit::Node &i) override;
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info) override;
    virtual void   execute() override;
};


//-----------------------------------------------------------------------------
class BFlowIso : public ::flow::Filter
{
public:
    BFlowIso() = default;
    virtual ~BFlowIso() {}

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
