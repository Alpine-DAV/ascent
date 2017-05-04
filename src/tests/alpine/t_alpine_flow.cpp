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
    r.add_entry("data",n,2);

    Node *n_fetch = r.fetch_entry("data");
    EXPECT_EQ(n,n_fetch);

    r.dec_entry_ref_count("data");
    n_fetch = r.fetch_entry("data");
    EXPECT_EQ(n,n_fetch);

    r.dec_entry_ref_count("data");
    
    n_fetch = r.fetch_entry("data");
    EXPECT_EQ(NULL,n_fetch);
    
}

//-----------------------------------------------------------------------------
class TestFilter: public flow::Filter
{
public:
    TestFilter()
    : flow::Filter()
    {
        Node &p = properties();
        p["type_name"] = "test";
        p["input_port_names"].append().set("in");
        p["default_params"]["value"] = "default";
        p["output_port"] = "true";

        init();
    }
        
    virtual ~TestFilter()
    {}

    virtual void execute()
    {
        Node *res = new Node();
        Node *in = input("in");
        res->set(*in);

        output().set(res);
    }
};


//-----------------------------------------------------------------------------
TEST(alpine_flow, alpine_flow_filter_graph)
{

    flow::FilterGraph f;
    
    f.add_filter("a", new TestFilter());
    f.add_filter("b", new TestFilter());
    f.add_filter("c", new TestFilter());
    
    f.connect("a","b","in");
    f.connect("b","c","in");
    
    
    f.print();

}

