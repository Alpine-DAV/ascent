//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_flow_registry.cpp
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
TEST(ascent_flow_registry, basic)
{
    Node *n = new Node();

    n->set(10);

    Registry r;
    r.add<Node>("d",n,2);
    r.print();

    Node *n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();


    n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();

    EXPECT_FALSE(r.has_entry("d"));

}

//-----------------------------------------------------------------------------
TEST(ascent_flow_registry, aliased)
{
    Node *n = new Node();

    n->set(10);


    Registry r;
    r.add<Node>("d1",n,1);
    r.add<Node>("d2",n,1);
    r.print();


    Node *n_fetch = r.fetch<Node>("d1");
    EXPECT_EQ(n,n_fetch);

    r.consume("d1");
    r.print();

    n_fetch = r.fetch<Node>("d2");
    EXPECT_EQ(n,n_fetch);

    r.consume("d2");
    r.print();

    EXPECT_FALSE(r.has_entry("d1"));

    EXPECT_FALSE(r.has_entry("d2"));

}


//-----------------------------------------------------------------------------
TEST(ascent_flow_registry, untracked)
{
    Node *n = new Node();

    n->set(10);

    Registry r;
    r.add<Node>("d",n,-1);
    r.print();

    Node *n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();

    n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);

    r.consume("d");
    r.print();


    n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);


    delete n;
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_registry, untracked_aliased)
{
    Node *n = new Node();

    n->set(10);

    Registry r;
    r.add<Node>("d",n,-1);
    r.add<Node>("d_al",n,1);
    r.print();

    Node *n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);
    r.consume("d");


    n_fetch = r.fetch<Node>("d_al");
    EXPECT_EQ(n,n_fetch);

    r.consume("d_al");

    EXPECT_FALSE(r.has_entry("d_al"));

    n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);

    r.print();

    delete n;
}


//-----------------------------------------------------------------------------
TEST(ascent_flow_registry, detach_tracked)
{
    Node *n = new Node();
    n->set(10);

    Registry r;
    r.add<Node>("d",n,1);
    r.print();

    Node *n_fetch = r.fetch<Node>("d");
    EXPECT_EQ(n,n_fetch);
    // liberate d
    r.detach("d");

    EXPECT_FALSE(r.has_entry("d"));

    r.print();

    delete n;
}



