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


