//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_workspace.cpp
///
//-----------------------------------------------------------------------------

#include "flow_workspace.hpp"
#include "flow_timer.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

// we init m_default_mpi_comm to -1, it's not clear if we can
// pick a safe non-inited value w/o the mpi headers, but
// we will try this strategy.
int Workspace::m_default_mpi_comm = -1;
static int g_timing_exec_count = 0;

//-----------------------------------------------------------------------------
class Workspace::ExecutionPlan
{
    public:

        static void generate(Graph &g,
                             conduit::Node &traversals);

    private:
        ExecutionPlan();
        ~ExecutionPlan();

        static void bf_topo_sort_visit(Graph &graph,
                                       const std::string &filter_name,
                                       conduit::Node &tags,
                                       conduit::Node &tarv);
};

//-----------------------------------------------------------------------------
class Workspace::FilterFactory
{
public:

    static std::map<std::string,FilterFactoryMethod> &registered_types()
    {
        return m_filter_types;
    }

private:
    static std::map<std::string,FilterFactoryMethod> m_filter_types;
};

//-----------------------------------------------------------------------------
std::map<std::string,FilterFactoryMethod> Workspace::FilterFactory::m_filter_types;


//-----------------------------------------------------------------------------
Workspace::ExecutionPlan::ExecutionPlan()
{
    // empty
}

//-----------------------------------------------------------------------------
Workspace::ExecutionPlan::~ExecutionPlan()
{
    // empty
}


//-----------------------------------------------------------------------------
void
Workspace::ExecutionPlan::generate(Graph &graph,
                                   conduit::Node &traversals)
{
    traversals.reset();

    Node snks;
    Node srcs;

    std::map<std::string,Filter*>::iterator itr;

    for(itr  = graph.m_filters.begin();
        itr != graph.m_filters.end();
        itr++)
    {
        Filter *f = itr->second;

        // check for snk
        if( !f->output_port() ||
             graph.edges_out(f->name()).number_of_children() == 0)
        {
            snks.append().set(f->name());
        }

        // check for src
        if( f->output_port() &&
            !graph.edges()["in"].has_child(f->name()) )
        {
            srcs.append().set(f->name());
        }


    }

    // init tags
    Node tags;
    for(itr  = graph.m_filters.begin();
        itr != graph.m_filters.end() ;
        itr++)
    {
        Filter *f = itr->second;
        tags[f->name()].set_int32(0);

    }

    // execute bf traversal from each snk

    NodeConstIterator snk_itr(&snks);
    while(snk_itr.has_next())
    {
        std::string snk_name = snk_itr.next().as_string();

        Node snk_trav;
        bf_topo_sort_visit(graph, snk_name, tags, snk_trav);
        if(snk_trav.number_of_children() > 0)
        {
            traversals.append().set(snk_trav);
        }
    }
}


//-----------------------------------------------------------------------------
void
Workspace::ExecutionPlan::bf_topo_sort_visit(Graph &graph,
                                             const std::string &f_name,
                                             conduit::Node &tags,
                                             conduit::Node &trav)
{
    if( tags[f_name].as_int32() != 0 )
    {
        return;
    }

    int uref = 1;
    tags[f_name].set_int32(1);

    Filter *f = graph.m_filters[f_name];

    if(f->output_port())
    {
        int num_refs = graph.edges_out(f_name).number_of_children();
        uref = num_refs > 0 ? num_refs : 1;
    }

    if ( f->port_names().number_of_children() > 0 )
    {
        NodeConstIterator f_inputs(&graph.edges_in(f_name));

        while(f_inputs.has_next())
        {
            const Node &n_f_input = f_inputs.next();

            if(n_f_input.dtype().is_string())
            {
                std::string f_in_name = n_f_input.as_string();
                bf_topo_sort_visit(graph, f_in_name, tags, trav);
            }
            else //  missing input.
            {
                index_t port_idx = f_inputs.index();
                CONDUIT_ERROR("Filter " << f->detailed_name()
                              << " is missing connection to input port "
                              << port_idx
                              << " ("
                              << f->port_index_to_name(port_idx)
                              << ")");
                uref = 0;
            }
        }
    }

    // conduit nodes keep insert order, so we can use
    // obj instead of list
    if(uref > 0)
    {
        trav[f_name] = uref;
    }

}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Workspace::Workspace()
:m_graph(this),
 m_registry(),
 m_timing_info(),
 m_enable_timings(false)
{

}

//-----------------------------------------------------------------------------
Workspace::~Workspace()
{

}

//-----------------------------------------------------------------------------
Graph &
Workspace::graph()
{
    return m_graph;
}

//-----------------------------------------------------------------------------
const Graph &
Workspace::graph() const
{
    return m_graph;
}


//-----------------------------------------------------------------------------
Registry &
Workspace::registry()
{
    return m_registry;
}

//-----------------------------------------------------------------------------
const Registry &
Workspace::registry() const
{
    return m_registry;
}

//-----------------------------------------------------------------------------
void
Workspace::traversals(Node &traversals)
{
    traversals.reset();
    ExecutionPlan::generate(graph(),traversals);
}

//-----------------------------------------------------------------------------
void
Workspace::execute()
{
    Timer t_total_exec;
    Node traversals;
    ExecutionPlan::generate(graph(),traversals);
    // execute traversals
    NodeIterator travs_itr = traversals.children();

    while(travs_itr.has_next())
    {
        NodeIterator trav_itr(&travs_itr.next());

        while(trav_itr.has_next())
        {
            Node &t = trav_itr.next();

            std::string  f_name = trav_itr.name();
            int          uref   = t.to_int32();
            Filter      *f      = graph().filters()[f_name];

            f->reset_inputs_and_output();

            // fetch inputs from reg, attach to filter's ports
            NodeConstIterator ports_itr = NodeConstIterator(&f->port_names());
            //registry().print();
            std::vector<std::string> f_i_names;
            while(ports_itr.has_next())
            {
                std::string port_name = ports_itr.next().as_string();
                std::string f_input_name = graph().edges_in(f_name)[port_name].as_string();
                f->set_input(port_name,&registry().fetch(f_input_name));
            }

            Timer t_flt_exec;
            // execute
            f->execute();

            if(m_enable_timings)
            {
                m_timing_info << g_timing_exec_count
                              << " " << f->name()
                              << " " << std::fixed << t_flt_exec.elapsed()
                              <<"\n";
            }

            // if has output, set output
            if(f->output_port())
            {
                if(f->output().data_ptr() == NULL)
                {
                    CONDUIT_ERROR("filter output is NULL, was set_output() called?");
                }

                registry().add(f_name,
                               f->output(),
                               uref);
            }

            f->reset_inputs_and_output();

            // consume inputs
            ports_itr.to_front();
            while(ports_itr.has_next())
            {
                std::string port_name = ports_itr.next().as_string();
                std::string f_input_name = graph().edges_in(f_name)[port_name].as_string();
                registry().consume(f_input_name);
            }
        }
    }

    if(m_enable_timings)
    {
        m_timing_info << g_timing_exec_count
                      << " [total] "
                      << std::fixed << t_total_exec.elapsed()
                      <<"\n";
        g_timing_exec_count++;
    }



}
//-----------------------------------------------------------------------------

void Workspace::enable_timings(bool enabled)
{
  m_enable_timings = enabled;
}

//-----------------------------------------------------------------------------
void
Workspace::reset()
{
    graph().reset();
    registry().reset();
}


//-----------------------------------------------------------------------------
void
Workspace::info(Node &out) const
{
    out.reset();

    graph().info(out["graph"]);
    registry().info(out["registry"]);
    out["timings"] = timing_info();
}


//-----------------------------------------------------------------------------
std::string
Workspace::to_json() const
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Workspace::print() const
{
    CONDUIT_INFO(to_json());
}

//-----------------------------------------------------------------------------
void
Workspace::reset_timing_info()
{
    g_timing_exec_count = 0;
    m_timing_info.str("");
}
//-----------------------------------------------------------------------------
string
Workspace::timing_info() const
{
    return m_timing_info.str();
}

//-----------------------------------------------------------------------------
Filter *
Workspace::create_filter(const std::string &filter_type_name)
{
    if(!supports_filter_type(filter_type_name))
    {
        CONDUIT_WARN("Cannot create unknown filter type: "
                    << filter_type_name);
        return NULL;
    }

    return FilterFactory::registered_types()[filter_type_name](filter_type_name.c_str());
}

//-----------------------------------------------------------------------------
void
Workspace::set_default_mpi_comm(int mpi_comm_id)
{
    m_default_mpi_comm = mpi_comm_id;
}

int
Workspace::default_mpi_comm()
{
    // we init m_default_mpi_comm to -1, it's not clear if we can
    // pick a safe non-inited value w/o the mpi headers, but
    // we will try this strategy.

    if(m_default_mpi_comm == -1)
    {
        CONDUIT_ERROR("flow::Workspace default MPI communicator is not initialized.")
    }

    return m_default_mpi_comm;
}


//-----------------------------------------------------------------------------
bool
Workspace::supports_filter_type(const std::string &filter_type_name)
{
    std::map<std::string,FilterFactoryMethod>::const_iterator itr;
    itr = FilterFactory::registered_types().find(filter_type_name);
    return (itr != FilterFactory::registered_types().end());
}

//-----------------------------------------------------------------------------
bool
Workspace::supports_filter_type(FilterFactoryMethod fr)
{
    Filter *f = fr("");

    Node iface;
    std::string f_type_name = "(type_name missing!)";
    f->declare_interface(iface);
    delete f;

    if(iface.has_child("type_name") &&
       iface["type_name"].dtype().is_string())
    {
        f_type_name = iface["type_name"].as_string();
    }

    return supports_filter_type(f_type_name);
}


//-----------------------------------------------------------------------------
void
Workspace::remove_filter_type(const std::string &filter_type_name)
{
    std::map<std::string,FilterFactoryMethod>::const_iterator itr;
    itr = FilterFactory::registered_types().find(filter_type_name);
    if(itr != FilterFactory::registered_types().end())
    {
        FilterFactory::registered_types().erase(filter_type_name);
    }
}

//-----------------------------------------------------------------------------
void
Workspace::register_filter_type(FilterFactoryMethod fr)
{
    if(supports_filter_type(fr))
    {
        // already registered
        return;
    }

    // obtain type name

    // check that filter is valid by creating
    // an instance
    Filter *f = fr("");

    // verify f provides proper interface declares

    Node i_test;
    Node v_info;

    std::string f_type_name = "(type_name missing!)";

    f->declare_interface(i_test);
    if(!Filter::verify_interface(i_test,v_info))
    {
        // if  the type name was provided, that helps improve
        // the error message, so try to include it
        if(i_test.has_child("type_name") &&
           i_test["type_name"].dtype().is_string())
        {
            f_type_name = i_test["type_name"].as_string();
        }

        // failed interface verify ...
        CONDUIT_ERROR("filter type interface verify failed." << std::endl
                      << f_type_name   << std::endl
                      << "Details:" << std::endl
                      << v_info.to_json());
    }

    f_type_name =i_test["type_name"].as_string();

    // we no longer need this instance ...
    delete f;

    register_filter_type(f_type_name,fr);
}


//-----------------------------------------------------------------------------
void
Workspace::register_filter_type(const std::string &filter_type_name,
                                FilterFactoryMethod fr)
{
    if(supports_filter_type(filter_type_name))
    {
        CONDUIT_INFO("filter type named:"
                      << filter_type_name
                      << " is already registered");
        return;
    }

    // check that filter is valid by creating
    // an instance

    Filter *f = fr(filter_type_name.c_str());

    // verify f provides proper interface declares

    Node i_test;
    Node v_info;

    std::string f_type_name = "(type_name missing!)";

    f->declare_interface(i_test);
    if(!Filter::verify_interface(i_test,v_info))
    {
        // if  the type name was provided, that helps improve
        // the error message, so try to include it
        if(i_test.has_child("type_name") &&
           i_test["type_name"].dtype().is_string())
        {
            f_type_name = i_test["type_name"].as_string();
        }

        // failed interface verify ...
        CONDUIT_ERROR("filter type interface verify failed." << std::endl
                      << f_type_name   << std::endl
                      << "Details:" << std::endl
                      << v_info.to_json());
    }

    f_type_name =i_test["type_name"].as_string();

    // we no longer need this instance ...
    delete f;

    if(supports_filter_type(f_type_name))
    {
        CONDUIT_ERROR("filter type named:"
                     << f_type_name
                    << " is already registered");
    }

    FilterFactory::registered_types()[filter_type_name] = fr;
}


//-----------------------------------------------------------------------------
std::string
Workspace::filter_type_name(FilterFactoryMethod fr)
{
    Filter *f = fr("");

    Node iface;
    std::string f_type_name = "(type_name missing!)";
    f->declare_interface(iface);
    delete f;

    if(iface.has_child("type_name") &&
       iface["type_name"].dtype().is_string())
    {
        f_type_name = iface["type_name"].as_string();
    }

    if(!supports_filter_type(f_type_name))
    {
        // TODO ERROR
    }

    return f_type_name;
}


//-----------------------------------------------------------------------------
void
Workspace::clear_supported_filter_types()
{
    FilterFactory::registered_types().clear();
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------




