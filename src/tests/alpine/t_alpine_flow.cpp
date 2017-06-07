//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: t_alpine_flow_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <alpine_flow.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"
#include "t_alpine_test_utils.hpp"



using namespace std;
using namespace conduit;
using namespace alpine;


//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_registry)
{
    Node *n = new Node();
    
    n->set(10);
    
    flow::Registry r;
    r.add("d",flow::Data(n),2);
    r.print();

    Node *n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();

    
    n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();
    
    EXPECT_FALSE(r.has_entry("d"));

}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_registry_aliased)
{
    Node *n = new Node();
    
    n->set(10);
    
    
    flow::Registry r;
    r.add("d1",flow::Data(n),1);
    r.add("d2",flow::Data(n),1);
    r.print();


    Node *n_fetch = r.fetch("d1");
    EXPECT_EQ(n,n_fetch);

    r.consume("d1");
    r.print();

    n_fetch = r.fetch("d2");
    EXPECT_EQ(n,n_fetch);

    r.consume("d2");
    r.print();

    EXPECT_FALSE(r.has_entry("d1"));
            
    EXPECT_FALSE(r.has_entry("d2"));

}


//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_registry_untracked)
{
    Node *n = new Node();
    
    n->set(10);

    flow::Registry r;
    r.add("d",flow::Data(n),-1);
    r.print();

    Node *n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();

    n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();

    
    n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);


    delete n;
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_registry_untracked_aliased)
{
    Node *n = new Node();
    
    n->set(10);

    flow::Registry r;
    r.add("d",flow::Data(n),-1);
    r.add("d_al",flow::Data(n),1);
    r.print();

    Node *n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);
    r.consume("d");


    n_fetch = r.fetch("d_al");
    EXPECT_EQ(n,n_fetch);

    r.consume("d_al");

    EXPECT_FALSE(r.has_entry("d_al"));

    n_fetch = r.fetch("d");
    EXPECT_EQ(n,n_fetch);

    r.print();
    
    delete n;
}

//-----------------------------------------------------------------------------
class SrcFilter: public flow::Filter
{
public:
    SrcFilter()
    : flow::Filter()
    {
        Node &i = interface();
        i["type_name"]   = "src";
        i["output_port"] = "true";
        i["port_names"] = DataType::empty();
        i["default_params"]["value"].set((int)0);

    }
        
    virtual ~SrcFilter()
    {}

    virtual void execute()
    {
        int val = params()["value"].value();

        // set output
        Node *res = new Node();
        res->set(val);
        output().set(res);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // bare-bones factory method
    static Filter *create()
    {
        return new SrcFilter();
    }
};


//-----------------------------------------------------------------------------
class IncFilter: public flow::Filter
{
public:
    IncFilter()
    : flow::Filter()
    {
        Node &i = interface();
        i["type_name"]   = "inc";
        i["output_port"] = "true";
        i["port_names"].append().set("in");
        i["default_params"]["inc"].set((int)1);
    }

    virtual ~IncFilter()
    {}

    virtual void execute()
    {
        
        // read in input param
        int inc  = params()["inc"].value();
        
        // get input data
        Node &in = input("in");
        int val  = in.to_int();
     
         // do something useful
        val+= inc;

        // set output 
        Node *res = new Node();
        res->set(val);
        
        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        
        output().set(res);

        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // bare-bones factory method
    static Filter *create()
    {
        return new IncFilter();
    }
};


//-----------------------------------------------------------------------------
class AddFilter: public flow::Filter
{
public:
    AddFilter()
    : flow::Filter()
    {
        Node &i = interface();
        i["type_name"]   = "add";
        i["output_port"] = "true";
        i["port_names"].append().set("a");
        i["port_names"].append().set("b");

    }
        
    virtual ~AddFilter()
    {}

    virtual void execute()
    {
        // grab data from inputs
        
        Node &a_in = input("a");
        Node &b_in = input("b");
        
        // do something useful 
        int rval = a_in.to_int() + b_in.to_int();
        
        // set output
        Node *res = new Node();
        res->set(rval);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        output().set(res);

        
        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // bare-bones factory method
    static Filter *create()
    {
        return new AddFilter();
    }
};




//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_linear)
{

    flow::Workspace w;
    // w.graph().register(FilterType(class))

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&IncFilter::create);


    w.graph().add_filter("src","s");
    
    w.graph().add_filter("inc","a");
    w.graph().add_filter("inc","b");
    w.graph().add_filter("inc","c");
    
    // // src, dest, port
    w.graph().connect("s","a","in");
    w.graph().connect("a","b","in");
    w.graph().connect("b","c","in");
    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("c");
    
    ALPINE_INFO("Final result: " << res.to_json());

    EXPECT_EQ(res.as_int(),3);

    w.registry().consume("c");

    w.print();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_linear_filter_ptr_iface)
{

    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&IncFilter::create);


    flow::Filter *f_s = w.graph().add_filter("src","s");
    
    flow::Filter *f_a = w.graph().add_filter("inc","a");
    flow::Filter *f_b = w.graph().add_filter("inc","b");
    flow::Filter *f_c = w.graph().add_filter("inc","c");
    
    f_a->connect_input_port("in",f_s);
    f_b->connect_input_port("in",f_a);
    f_c->connect_input_port("in",f_b);
    
    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("c");
    
    ALPINE_INFO("Final result: " << res.to_json());

    EXPECT_EQ(res.as_int(),3);

    w.registry().consume("c");

    w.print();
}


//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_linear_filter_ptr_iface_port_idx)
{

    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&IncFilter::create);


    flow::Filter *f_s = w.graph().add_filter("src","s");
    
    flow::Filter *f_a = w.graph().add_filter("inc","a");
    flow::Filter *f_b = w.graph().add_filter("inc","b");
    flow::Filter *f_c = w.graph().add_filter("inc","c");
    
    f_a->connect_input_port(0,f_s);
    f_b->connect_input_port(0,f_a);
    f_c->connect_input_port(0,f_b);
    
    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("c");
    
    ALPINE_INFO("Final result: " << res.to_json());

    EXPECT_EQ(res.as_int(),3);

    w.registry().consume("c");

    w.print();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_graph)
{
    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&AddFilter::create);


    Node p_vs;
    p_vs["value"].set(int(10));

    w.graph().add_filter("src","v1",p_vs);
    w.graph().add_filter("src","v2",p_vs);
    w.graph().add_filter("src","v3",p_vs);
    
    
    w.graph().add_filter("add","a1");
    w.graph().add_filter("add","a2");
    
    
    // // src, dest, port
    w.graph().connect("v1","a1","a");
    w.graph().connect("v2","a1","b");
    
    
    w.graph().connect("a1","a2","a");
    w.graph().connect("v3","a2","b");

    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("a2");
    
    ALPINE_INFO("Final result: " << res.to_json());
    
    EXPECT_EQ(res.as_int(),30);
    
    w.registry().consume("a2");
    
    w.print();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_graph_filter_ptr_iface)
{
    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&AddFilter::create);


    Node p_vs;
    p_vs["value"].set(int(10));

    flow::Filter *f_v1 = w.graph().add_filter("src","v1",p_vs);
    flow::Filter *f_v2 = w.graph().add_filter("src","v2",p_vs);
    flow::Filter *f_v3 = w.graph().add_filter("src","v3",p_vs);
    
    
    flow::Filter *f_a1 = w.graph().add_filter("add","a1");
    flow::Filter *f_a2 = w.graph().add_filter("add","a2");

    
    f_a1->connect_input_port("a",f_v1);
    f_a1->connect_input_port("b",f_v2);

    f_a2->connect_input_port("a",f_a1);
    f_a2->connect_input_port("b",f_v3);

    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("a2");
    
    ALPINE_INFO("Final result: " << res.to_json());
    
    EXPECT_EQ(res.as_int(),30);
    
    w.registry().consume("a2");
    
    w.print();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_graph_filter_ptr_iface_port_idx)
{
    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&AddFilter::create);


    Node p_vs;
    p_vs["value"].set(int(10));

    flow::Filter *f_v1 = w.graph().add_filter("src","v1",p_vs);
    flow::Filter *f_v2 = w.graph().add_filter("src","v2",p_vs);
    flow::Filter *f_v3 = w.graph().add_filter("src","v3",p_vs);
    
    
    flow::Filter *f_a1 = w.graph().add_filter("add","a1");
    flow::Filter *f_a2 = w.graph().add_filter("add","a2");

    
    f_a1->connect_input_port(0,f_v1);
    f_a1->connect_input_port(1,f_v2);

    f_a2->connect_input_port(0,f_a1);
    f_a2->connect_input_port(1,f_v3);

    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("a2");
    
    ALPINE_INFO("Final result: " << res.to_json());
    
    EXPECT_EQ(res.as_int(),30);
    
    w.registry().consume("a2");
    
    w.print();
}

//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_reg_source)
{

    flow::Workspace w;

    w.graph().register_filter_type(&flow::filters::RegistrySource::create);
    w.graph().register_filter_type(&AddFilter::create);


    Node v;
    
    v.set(int(10));

    Node p;
    p["entry"] = ":src";
    w.registry().add(":src",flow::Data(&v));
    
    w.graph().add_filter("registry_source","s",p);
    
    w.graph().add_filter("add","a");

    // // src, dest, port
    w.graph().connect("s","a","a");
    w.graph().connect("s","a","b");

    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch("a");
    
    ALPINE_INFO("Final result: " << res.to_json());
    
    EXPECT_EQ(res.as_int(),20);
    
    w.registry().consume("a");
    
    w.print();
    
    Node *n_s = w.registry().fetch(":src");
    
    ALPINE_INFO("Input result: " << n_s->to_json());
    
    EXPECT_EQ(n_s->as_int(),10);
    
    EXPECT_EQ(n_s,&v);
}


//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_graph_filter_ptr_iface_auto_name)
{
    flow::Workspace w;

    w.graph().register_filter_type(&SrcFilter::create);
    w.graph().register_filter_type(&AddFilter::create);


    Node p_vs;
    p_vs["value"].set(int(10));

    flow::Filter *f_v1 = w.graph().add_filter("src",p_vs);
    flow::Filter *f_v2 = w.graph().add_filter("src",p_vs);
    flow::Filter *f_v3 = w.graph().add_filter("src",p_vs);
    
    
    
    flow::Filter *f_a1 = w.graph().add_filter("add");
    flow::Filter *f_a2 = w.graph().add_filter("add");


    EXPECT_EQ(f_v1->name(),"f_0");
    EXPECT_EQ(f_v2->name(),"f_1");
    EXPECT_EQ(f_v3->name(),"f_2");
    EXPECT_EQ(f_a1->name(),"f_3");
    EXPECT_EQ(f_a2->name(),"f_4");

    
    f_a1->connect_input_port("a",f_v1);
    f_a1->connect_input_port("b",f_v2);

    f_a2->connect_input_port("a",f_a1);
    f_a2->connect_input_port("b",f_v3);

    //
    w.print();
    //
    w.execute();
    
    Node &res = w.registry().fetch(f_a2->name());
    
    ALPINE_INFO("Final result: " << res.to_json());
    
    EXPECT_EQ(res.as_int(),30);
    
    w.registry().consume(f_a2->name());
    
    w.print();
}


//-----------------------------------------------------------------------------
// TEST(alpine_flow, alpine_flow_workspace_notes)
// {

    // flow::Workspace w;

    
    // FILTER_FACTORY(TestFilter) ?
    // w.graph().register_filter_type(&TestFilter::type);
    //
    // Node n;
    // n.set(64);
    //
    // w.add_source(":src",&n);
    // // does the following:
    //
    // // adds with ref count = -1, registry won't reap
    // w.registry().add_entry(":src",&n);
    // w.graph().add_filter("registry_source",":src");
    //
    //
    // //
    //
    // w.graph().add_filter("test","a");
    // w.graph().add_filter("test","b");
    // w.graph().add_filter("test","c");
    //
    // ///
    //
    // // src, dest, port
    // w.graph().connect("a","b","in");
    // w.graph().connect("b","c","in");
    //
    // // special case, one input port
    // w.graph().connect("a","b");
    // w.graph().connect("b","c");
    //
    //
    // w.print();
    //
    // // supports auto name ... (f%04d % count)
    // Filter &f0 = w.graph().add_filter("test");
    // Filter &f1 = w.graph().add_filter("test");
    // Filter &f2 = w.graph().add_filter("test");
    //
    // f0.name() == "f0000";
    // f1.name() == "f0001";
    // f2.name() == "f0002";
    //
    //
    // // dest.connect(src) ?
    // // dest.connect(src, port) ?
    // f1.connect(f0,"in")
    // f2.connect(f1,"in")
    //
    //
    // Filter &fsnk = w.add_sink(":dest");
    // // does the following:
    // w.graph().add_filter("registry_sink",":dest");
    //
    // fsnk.connect(f2);
    

// }


