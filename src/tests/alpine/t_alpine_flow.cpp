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
    
    
    // TODO, we want n to be deleted ...
    
    
    n_fetch = r.fetch("d");
    EXPECT_EQ(NULL,n_fetch);
    
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

    
    // TODO, we want n to be deleted ...    
    
    n_fetch = r.fetch("d1");
    EXPECT_EQ(NULL,n_fetch);

    n_fetch = r.fetch("d2");
    EXPECT_EQ(NULL,n_fetch);

    
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

    n_fetch = r.fetch("d_al");
    EXPECT_EQ(NULL,n_fetch);
    

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
        Node &p = properties();
        p["type_name"] = "src";
        p["port_names"] = DataType::empty();
        p["default_params"]["value"].set((int)0);
        p["output_port"] = "true";
    }
        
    virtual ~SrcFilter()
    {}

    virtual void execute()
    {;
        int val = params()["value"].value();
        Node *res = new Node();
        res->set(val);
        output().set(res);
        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // stand in for factory solution
    static Filter *type()
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
        Node &p = properties();
        p["type_name"] = "inc";
        p["port_names"].append().set("in");
        p["default_params"]["inc"].set((int)1);
        p["output_port"] = "true";
    }
        
    virtual ~IncFilter()
    {}

    virtual void execute()
    {
        Node &in = input("in");
        int val  = in.to_int();
        int inc  = params()["inc"].value();

        val+= inc;

        Node *res = new Node();
        res->set(val);
        
        output().set(res);
        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // stand in for factory solution
    static Filter *type()
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
        Node &p = properties();
        p["type_name"] = "add";
        p["port_names"].append().set("a");
        p["port_names"].append().set("b");
        p["output_port"] = "true";
    }
        
    virtual ~AddFilter()
    {}

    virtual void execute()
    {
        Node &a_in = input("a");
        Node &b_in = input("b");
        
        int rval = a_in.to_int() + b_in.to_int();
        Node *res = new Node();
        res->set(rval);
        output().set(res);
        
        ALPINE_INFO("exec: " << name() << " result = " << res->to_json());
    }

    // stand in for factory solution
    static Filter *type()
    {
        return new AddFilter();
    }
};




//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_workspace_linear)
{

    flow::Workspace w;
    // w.graph().register(FilterType(class))

    w.graph().register_filter_type(&SrcFilter::type);    
    w.graph().register_filter_type(&IncFilter::type);


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
TEST(alpine_flow, alpine_flow_workspace_graph)
{

    flow::Workspace w;
    // w.graph().register(FilterType(class))

    w.graph().register_filter_type(&SrcFilter::type);
    w.graph().register_filter_type(&AddFilter::type);


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
TEST(alpine_flow, alpine_flow_workspace_reg_source)
{

    flow::Workspace w;
    // w.graph().register(FilterType(class))

    w.graph().register_filter_type(&flow::filters::RegistrySource::type);
    w.graph().register_filter_type(&AddFilter::type);


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


