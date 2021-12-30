//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_flow_data.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <flow.hpp>

#include <iostream>
#include <math.h>

#include "t_config.hpp"



using namespace std;
using namespace conduit;
using namespace flow;


//-----------------------------------------------------------------------------
TEST(ascent_flow_data, generic_wrap)
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
// dummy class used to track and test ascent::Data release imp
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
TEST(ascent_flow_data, generic_release)
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




