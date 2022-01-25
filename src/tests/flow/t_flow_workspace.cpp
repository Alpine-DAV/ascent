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
class SrcFilter: public Filter
{
public:
    SrcFilter()
    : Filter()
    {}

    virtual ~SrcFilter()
    {}


    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "src";
        i["output_port"] = "true";
        i["port_names"] = DataType::empty();
        i["default_params"]["value"].set((int)0);
    }


    virtual void execute()
    {
        int val = params()["value"].value();

        // set output
        Node *res = new Node();
        res->set(val);
        set_output<Node>(res);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        ASCENT_INFO("exec: " << name() << " result = " << res->to_json());
    }
};


//-----------------------------------------------------------------------------
class IncFilter: public Filter
{
public:
    IncFilter()
    : Filter()
    {}

    virtual ~IncFilter()
    {}

    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "inc";
        i["output_port"] = "true";
        i["port_names"].append().set("in");
        i["default_params"]["inc"].set((int)1);
    }

    virtual void execute()
    {

        // read in input param
        int inc  = params()["inc"].value();

        // get input data


        Node *in = input<Node>("in");
        int val  = in->to_int();

         // do something useful
        val+= inc;

        // set output
        Node *res = new Node();
        res->set(val);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.

        set_output<Node>(res);

        ASCENT_INFO("exec: " << name() << " result = " << res->to_json());
    }

};


//-----------------------------------------------------------------------------
class AddFilter: public Filter
{
public:
    AddFilter()
    : Filter()
    {}

    virtual ~AddFilter()
    {}

    virtual void declare_interface(Node &i)
    {
        i["type_name"]   = "add";
        i["output_port"] = "true";
        i["port_names"].append().set("a");
        i["port_names"].append().set("b");
    }

    virtual void execute()
    {
        // grab data from inputs

        Node *a_in = input<Node>("a");
        Node *b_in = input<Node>("b");

        // do something useful
        int rval = a_in->to_int() + b_in->to_int();

        // set output
        Node *res = new Node();
        res->set(rval);

        // the registry will take care of deleting the data
        // when all consuming filters have executed.
        set_output<Node>(res);


        ASCENT_INFO("exec: " << name() << " result = " << res->to_json());
    }

};




//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, linear_graph)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<IncFilter>();

    Workspace w;
    // w.graph().register(FilterType(class))

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

    Node *res = w.registry().fetch<Node>("c");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),3);

    w.registry().consume("c");

    w.print();

    ASCENT_INFO(w.timing_info());

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, linear_graph_using_filter_ptr_iface)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<IncFilter>();

    Workspace w;

    Filter *f_s = w.graph().add_filter("src","s");

    Filter *f_a = w.graph().add_filter("inc","a");
    Filter *f_b = w.graph().add_filter("inc","b");
    Filter *f_c = w.graph().add_filter("inc","c");

    //w.graph().connect("s","a","in");
    f_a->connect_input_port("in",f_s);
    // w.graph().connect("a","b","in");
    f_b->connect_input_port("in",f_a);
    // w.graph().connect("b","c","in");
    f_c->connect_input_port("in",f_b);

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("c");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),3);

    w.registry().consume("c");

    w.print();

    // test various graph rep output helpers

    std::string r_json = w.graph().to_json();
    std::string r_yaml = w.graph().to_yaml();

    ASCENT_INFO("Graph JSON" << r_json);
    ASCENT_INFO("Graph YAML" << r_yaml);
    
    // make sure we can parse back with conduit
    
    Node n_parse_test;
    Generator g_json(r_json,"json");
    g_json.walk(n_parse_test);
    EXPECT_EQ(r_json,n_parse_test.to_json());

    Generator g_yaml(r_yaml,"yaml");
    g_yaml.walk(n_parse_test);
    EXPECT_EQ(r_yaml,n_parse_test.to_yaml());
    
    ASCENT_INFO("Graph Dot" 
                << w.graph().to_dot());

    ASCENT_INFO("Graph Dot HTML" 
                << w.graph().to_dot_html());

    string output_path = prepare_output_dir();
    string output_file_dot_txt  = conduit::utils::join_file_path(output_path,
                                         "tout_flow_workspace_linear_dot.txt");
    string output_file_dot_html = conduit::utils::join_file_path(output_path,
                                        "tout_flow_workspace_linear_dot.html");

    // remove old files
    if(conduit::utils::is_file(output_file_dot_txt))
    {
        conduit::utils::remove_file(output_file_dot_txt);
    }

    if(conduit::utils::is_file(output_file_dot_html))
    {
        conduit::utils::remove_file(output_file_dot_html);
    }

    w.graph().save_dot(output_file_dot_txt);
    w.graph().save_dot_html(output_file_dot_html);

    EXPECT_TRUE(conduit::utils::is_file(output_file_dot_txt));
    EXPECT_TRUE(conduit::utils::is_file(output_file_dot_html));

    Workspace::clear_supported_filter_types();
}


//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, linear_graph_using_filter_ptr_iface_and_port_idx)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<IncFilter>();

    Workspace w;


    Filter *f_s = w.graph().add_filter("src","s");

    Filter *f_a = w.graph().add_filter("inc","a");
    Filter *f_b = w.graph().add_filter("inc","b");
    Filter *f_c = w.graph().add_filter("inc","c");

    f_a->connect_input_port(0,f_s);
    f_b->connect_input_port(0,f_a);
    f_c->connect_input_port(0,f_b);

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("c");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),3);

    w.registry().consume("c");

    w.print();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow, ascent_flow_workspace_graph)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();


    Workspace w;

    Node p_vs;
    p_vs["value"].set(int(10));

    w.graph().add_filter("src","v1",p_vs);
    w.graph().add_filter("src","v2",p_vs);
    w.graph().add_filter("src","v3",p_vs);


    w.graph().add_filter("add","a1");
    w.graph().add_filter("add","a2");



    // ascii art pictures?

    // // src, dest, port
    w.graph().connect("v1","a1","a");
    w.graph().connect("v2","a1","b");


    w.graph().connect("a1","a2","a");
    w.graph().connect("v3","a2","b");

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("a2");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w.registry().consume("a2");

    w.print();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, dag_graph_filter_ptr_iface)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;


    Node p_vs;
    p_vs["value"].set(int(10));

    Filter *f_v1 = w.graph().add_filter("src","v1",p_vs);
    Filter *f_v2 = w.graph().add_filter("src","v2",p_vs);
    Filter *f_v3 = w.graph().add_filter("src","v3",p_vs);


    Filter *f_a1 = w.graph().add_filter("add","a1");
    Filter *f_a2 = w.graph().add_filter("add","a2");


    f_a1->connect_input_port("a",f_v1);
    f_a1->connect_input_port("b",f_v2);

    f_a2->connect_input_port("a",f_a1);
    f_a2->connect_input_port("b",f_v3);

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("a2");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w.registry().consume("a2");

    w.print();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, dag_graph_filter_ptr_iface_port_idx)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();


    Workspace w;

    Node p_vs;
    p_vs["value"].set(int(10));

    Filter *f_v1 = w.graph().add_filter("src","v1",p_vs);
    Filter *f_v2 = w.graph().add_filter("src","v2",p_vs);
    Filter *f_v3 = w.graph().add_filter("src","v3",p_vs);


    Filter *f_a1 = w.graph().add_filter("add","a1");
    Filter *f_a2 = w.graph().add_filter("add","a2");


    f_a1->connect_input_port(0,f_v1);
    f_a1->connect_input_port(1,f_v2);

    f_a2->connect_input_port(0,f_a1);
    f_a2->connect_input_port(1,f_v3);

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("a2");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w.registry().consume("a2");

    w.print();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, graph_workspace_reg_source)
{
    Workspace::register_filter_type<filters::RegistrySource>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;
    // create data
    Node v;
    v.set(int(10));
    // add to reg with name
    w.registry().add<Node>(":src",&v);

    // create a filter that allows us to access this as data
    Node p;
    p["entry"] = ":src";
    w.graph().add_filter("registry_source","s",p);

    w.graph().add_filter("add","a");

    // // src, dest, port
    w.graph().connect("s","a","a");
    w.graph().connect("s","a","b");

    //
    w.print();
    //
    w.execute();

    Node *res = w.registry().fetch<Node>("a");

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),20);

    w.registry().consume("a");

    w.print();

    Node *n_s = w.registry().fetch<Node>(":src");

    ASCENT_INFO("Input result: " << n_s->to_json());

    EXPECT_EQ(n_s->to_int(),10);

    EXPECT_EQ(n_s,&v);

    Workspace::clear_supported_filter_types();
}


//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, dag_graph_filter_ptr_iface_auto_name)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;


    Node p_vs;
    p_vs["value"].set(int(10));

    Filter *f_v1 = w.graph().add_filter("src",p_vs);
    Filter *f_v2 = w.graph().add_filter("src",p_vs);
    Filter *f_v3 = w.graph().add_filter("src",p_vs);



    Filter *f_a1 = w.graph().add_filter("add");
    Filter *f_a2 = w.graph().add_filter("add");


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

    Node *res = w.registry().fetch<Node>(f_a2->name());

    ASCENT_INFO("Final result: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w.registry().consume(f_a2->name());

    w.print();

    Workspace::clear_supported_filter_types();
}

//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, graph_info)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;

    Node p_vs;
    p_vs["value"].set(int(10));

    w.graph().add_filter("src","v1",p_vs);
    w.graph().add_filter("src","v2",p_vs);
    w.graph().add_filter("src","v3",p_vs);


    w.graph().add_filter("add","a1");
    w.graph().add_filter("add","a2");


    // ascii art pictures?

    // // src, dest, port
    w.graph().connect("v1","a1","a");
    w.graph().connect("v2","a1","b");


    w.graph().connect("a1","a2","a");
    w.graph().connect("v3","a2","b");

    //
    w.graph().print();
    w.print();

    ASCENT_INFO(w.graph().to_dot());

    Workspace::clear_supported_filter_types();
}


//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, dag_graph_save_and_load)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;

    Node p_vs;
    p_vs["value"].set(int(10));

    w.graph().add_filter("src","v1",p_vs);
    w.graph().add_filter("src","v2",p_vs);
    w.graph().add_filter("src","v3",p_vs);


    w.graph().add_filter("add","a1");
    w.graph().add_filter("add","a2");


    // ascii art pictures?

    // // src, dest, port
    w.graph().connect("v1","a1","a");
    w.graph().connect("v2","a1","b");


    w.graph().connect("a1","a2","a");
    w.graph().connect("v3","a2","b");

    //
    w.print();
    prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_dir(),
                                                        "tout_flow_wrkspace_dag_save_and_load.json");

    w.graph().save(output_file);

    w.execute();

    Node *res = w.registry().fetch<Node>("a2");

    ASCENT_INFO("Result: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    // load the graph into a new workspace and execute it
    Workspace w2;

    w2.graph().load(output_file);
    w2.print();
    w2.execute();
    res = w2.registry().fetch<Node>("a2");

    ASCENT_INFO("Result from loaded graph: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w2.registry().consume("a2");

    w2.print();

    // reload the graph and execute it
    w2.graph().load(output_file);

    Node w2_info;
    w2.info(w2_info);
    w2.print();
    w2.execute();

    res = w2.registry().fetch<Node>("a2");

    ASCENT_INFO("Result from loaded graph: " << res->to_json());

    EXPECT_EQ(res->to_int(),30);

    w2.registry().consume("a2");

    w2.print();

    Workspace w3;
    w3.graph().add_graph(w2_info["graph"]);
    w3.execute();
    res = w3.registry().fetch<Node>("a2");
    EXPECT_EQ(res->to_int(),30);
    w3.registry().consume("a2");
    w3.print();

    // check add_filters and add_connections
    w3.reset();
    w3.graph().add_filters(w2_info["graph/filters"]);
    w3.graph().add_connections(w2_info["graph/connections"]);
    w3.execute();
    res = w3.registry().fetch<Node>("a2");
    EXPECT_EQ(res->to_int(),30);
    w3.registry().consume("a2");
    w3.print();



    Workspace::clear_supported_filter_types();
}



//-----------------------------------------------------------------------------
TEST(ascent_flow_workspace, dag_graph_missing_input_error)
{
    Workspace::register_filter_type<SrcFilter>();
    Workspace::register_filter_type<AddFilter>();

    Workspace w;

    Node p_vs;
    p_vs["value"].set(int(10));

    w.graph().add_filter("src","v1",p_vs);
    w.graph().add_filter("src","v2",p_vs);
    w.graph().add_filter("src","v3",p_vs);


    w.graph().add_filter("add","a1");
    w.graph().add_filter("add","a2");


    // ascii art pictures?

    // // src, dest, port
    w.graph().connect("v1","a1","a");
    w.graph().connect("v2","a1","b");


    w.graph().connect("a1","a2","a");
    // omit connection to trigger error
    //w.graph().connect("v3","a2","b");

    //
    w.print();
    Node tout;
    EXPECT_THROW(w.traversals(tout),conduit::Error);
    EXPECT_THROW(w.execute(),conduit::Error);

    Workspace::clear_supported_filter_types();
}
