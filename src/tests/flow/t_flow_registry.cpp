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



