//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_flow_workspace.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <flow.hpp>
#include <flow_builtin_filters.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"
#include "t_utils.hpp"



using namespace std;
using namespace conduit;
using namespace ascent;
using namespace flow;

//-----------------------------------------------------------------------------
class Payload
{
public:
    Payload(int val)
    :m_val(val)
    {

    }
    ~Payload()
    {
        std::cout << "deleting payload" << std::endl;
    }

    void print()
    {
        std::cout << "I am a Payload, here is my value" << m_val << std::endl;
    }

    int m_val;
};


//-----------------------------------------------------------------------------
class Aggregator
{
public:
    Aggregator(flow::Registry *r)
    :m_agg_count(0),
     m_r(r)
    {

    }

    ~Aggregator()
    {
        std::cout << "deleting aggregator" << std::endl;
    }

    void add_payload(Payload *pl)
    {
        ostringstream oss;
        oss << "key_" << m_agg_count;
        m_r->add<Payload>(oss.str(),pl,1);
        m_agg_count++;
    }

    void exec()
    {
        for(int i=0; i < m_agg_count; i++)
        {
            ostringstream oss;
            oss << "key_" << i;
            Payload *pl = m_r->fetch<Payload>(oss.str());
            pl->print();
        }


        for(int i=0; i < m_agg_count; i++)
        {
            ostringstream oss;
            oss << "key_" << i;
            m_r->consume(oss.str());
        }
    }

    int m_agg_count;
    flow::Registry *m_r;

};

//-----------------------------------------------------------------------------
class CreatePayload: public Filter
{
public:
    CreatePayload()
    : Filter()
    {}

    virtual ~CreatePayload()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "create_payload";
        i["output_port"] = "true";
        i["port_names"] = DataType::empty();
    }

    virtual void execute()
    {
        set_output<Payload>(new Payload(42));
    }
};

//-----------------------------------------------------------------------------
class CreateAgg: public Filter
{
public:
    CreateAgg()
    : Filter()
    {}

    virtual ~CreateAgg()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "create_agg";
        i["output_port"] = "true";
        i["port_names"] = DataType::empty();
    }


    virtual void execute()
    {
        Aggregator *agg = new Aggregator(&graph().workspace().registry());
        set_output<Aggregator>(agg);
    }
};

//-----------------------------------------------------------------------------
class AddPayload: public Filter
{
public:
    AddPayload()
    : Filter()
    {}

    virtual ~AddPayload()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "add_payload";
        i["output_port"] = "true";
        i["port_names"].append() = "agg";
        i["port_names"].append() = "src";
    }


    virtual void execute()
    {
        Aggregator *agg = input<Aggregator>(0);
        Payload *pl = input<Payload>(1);
        agg->add_payload(pl);
        set_output<Aggregator>(agg);
    }
};


//-----------------------------------------------------------------------------
class ExecAgg: public Filter
{
public:
    ExecAgg()
    : Filter()
    {}

    virtual ~ExecAgg()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "exec_agg";
        i["output_port"] = "false";
        i["port_names"].append() = "agg";
    }

    virtual void execute()
    {
        Aggregator *agg = input<Aggregator>(0);
        agg->exec();
    }
};

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace_adv_manage, test_agg)
{
    Workspace::register_filter_type<CreatePayload>();
    Workspace::register_filter_type<AddPayload>();
    Workspace::register_filter_type<CreateAgg>();
    Workspace::register_filter_type<ExecAgg>();

    Workspace w;

    w.graph().add_filter("create_payload","pl_a");
    w.graph().add_filter("create_payload","pl_b");
    w.graph().add_filter("create_agg","agg");

    w.graph().add_filter("add_payload","add_pl_a");
    w.graph().add_filter("add_payload","add_pl_b");

    w.graph().add_filter("exec_agg","go");

    // // src, dest, port

    w.graph().connect("agg","add_pl_a",0);
    w.graph().connect("pl_a","add_pl_a",1);

    w.graph().connect("add_pl_a","add_pl_b",0);
    w.graph().connect("pl_b","add_pl_b",1);

    w.graph().connect("add_pl_b","go",0);

    //
    w.print();
    //
    w.execute();

}


