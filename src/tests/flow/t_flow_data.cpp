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
/// file: t_alpine_flow_data.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <alpine_flow.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"



using namespace std;
using namespace conduit;
using namespace alpine;
using namespace alpine::flow;


//-----------------------------------------------------------------------------
TEST(alpine_flow_data, generic_wrap)
{
    Node *n = new Node();
    
    n->set("test");
    
    DataWrapper<Node> n_wrap(n);
    
    Node *n_res = n_wrap.value<Node>();
    
    
    EXPECT_EQ(n_res->as_string(),"test");

    // runtime type checking of instance using a static type ...
    EXPECT_TRUE(n_wrap.check_type<Node>());
    EXPECT_FALSE(n_wrap.check_type<int>());

}

//-----------------------------------------------------------------------------
// dummy class used to track and test alpine::Data release imp
class TestData
{
public:
    
    TestData()
    {
        m_id = m_created;
        m_created++;
    }

    ~TestData()
    {
        m_destroyed++;
    }
    
    int id() {return m_id;}
    
    
    static void clear_stats() { m_created = 0; m_destroyed = 0;}
    
    static int created()  { return m_created;}
    static int destroyed(){ return m_destroyed;}
    
private:
           int m_id;
    static int m_created;
    static int m_destroyed;
};

int TestData::m_created   = 0;
int TestData::m_destroyed = 0;



//-----------------------------------------------------------------------------
TEST(alpine_flow_data, generic_release)
{
    TestData::clear_stats();
    
    TestData *t = new TestData();
    
    DataWrapper<TestData> t_wrap(t);
    
    TestData *t_res = t_wrap.value<TestData>();
        
    EXPECT_EQ(t_res->id(),0);
    
    EXPECT_EQ(TestData::created(),1);
    EXPECT_EQ(TestData::destroyed(),0);
    

    t_wrap.release();

    EXPECT_EQ(TestData::created(),1);
    EXPECT_EQ(TestData::destroyed(),1);

    
    t = new TestData();


    DataWrapper<TestData> t_wrap2(t);
    
    t_res = t_wrap2.value<TestData>();
        
    EXPECT_EQ(t_res->id(),1);
    
    EXPECT_EQ(TestData::created(),2);
    EXPECT_EQ(TestData::destroyed(),1);
    

    t_wrap2.release();

    EXPECT_EQ(TestData::created(),2);
    EXPECT_EQ(TestData::destroyed(),2);

}




