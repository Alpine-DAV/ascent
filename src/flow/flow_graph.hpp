//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_graph.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_GRAPH_HPP
#define FLOW_GRAPH_HPP

#include <flow_exports.h>
#include <flow_config.h>
#include <flow_filter.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

class Workspace;

//-----------------------------------------------------------------------------
///
/// Filter Graph
///
//-----------------------------------------------------------------------------

class FLOW_API Graph
{
public:

    friend class Workspace;
    friend class ExecutionPlan;

   ~Graph();

    /// access workspace that owns this graph
    Workspace &workspace();


    /// add a new filter of given type, and assigned name
    Filter *add_filter(const std::string &filter_type,
                       const std::string &name);


    /// add a new filter of given type, assigned name,
    /// and params
    Filter *add_filter(const std::string &filter_type,
                       const std::string &name,
                       const conduit::Node &params);

    /// add a new filter of given type, let the graph
    /// generate a unique a name
    Filter *add_filter(const std::string &filter_type);

    /// add a new filter of given type using params
    /// let the graph generate a unique a name
    Filter *add_filter(const std::string &filter_type,
                       const conduit::Node &params);

    /// connect src filter to dest's input port
    /// using portname
    void connect(const std::string &src_name,
                 const std::string &des_name,
                 const std::string &port_name);

    /// connect src filter to dest's input port
    /// using port index
    void connect(const std::string &src_name,
                 const std::string &des_name,
                 int port_idx);


    /// check if this graph has a filter with passed name
    bool has_filter(const std::string &name);

    /// remove if filter with passed name from this graph
    void remove_filter(const std::string &name);

    /// this methods are used by save() and info()
    /// the produce conduit trees with data that can be used
    /// add_filters() and add_connections().

    /// Provides a conduit description of the filters in the graph
    void filters(conduit::Node &out) const;
    /// Provides a conduit description of the connections in the graph
    void connections(conduit::Node &out) const;

    /// adds a set of filters from a conduit tree that describes them
    void add_filters(const conduit::Node &filters);
    /// adds a set of connections from a conduit tree that describes them
    void add_connections(const conduit::Node &conns);


    /// adds a set of filters and connections from the given graph
    void add_graph(const Graph &g);
    /// adds a set of filters and connections from a conduit tree that
    //  describes them
    void add_graph(const conduit::Node &g);


    /// remove all filters
    void reset();

    /// save graph graph state to a conduit tree,
    /// which can be used to restore the graph with load
    void save(conduit::Node &n);

    /// save graph graph state to a file,
    /// which can be used to restore the graph with load
    void save(const std::string &ofile,
              const std::string &protocol="conduit_json");

    /// load graph from file
    ///  equiv to:
    ///   load n from ofile
    ///   reset();
    ///   add_filters(n["filters"]);
    ///   add_connections(n["connections"]);
    /// (Note: does not handle filter type registration)
    void load(const std::string &ofile,
              const std::string &protocol="conduit_json");

    /// load graph from conduit tree
    ///  equiv to:
    ///   reset();
    ///   add_filters(n["filters"]);
    ///   add_connections(n["connections"]);
    /// (Note: does not handle filter type registration)
    void load(const conduit::Node &n);

    /// create human understandable tree that describes the state
    /// of the workspace
    void           info(conduit::Node &out) const;
    /// create json string from info
    std::string    to_json() const;
    /// create yaml string from info
    std::string    to_yaml() const;
    /// print json version of info
    void           print() const;

    /// graphviz output helpers
    
    /// create dot description
    std::string to_dot() const;
    /// create dot description embedded d3 + html
    std::string to_dot_html() const;

    /// stream variants of graphviz output helpers
    void to_dot(std::ostream &oss,
                const std::string &eol = "\n") const;
    void to_dot_html(std::ostream &oss) const;

    /// save graphviz output to txt file
    void save_dot(const std::string &ofile) const;

    /// save graphviz output to a d3 + html output, viewable
    /// in a web browser.
    void save_dot_html(const std::string &ofile) const;

private:
    Graph(Workspace *w);

    void                 init();

    const conduit::Node &edges()  const;
    const conduit::Node &edges_in(const std::string &f_name)  const;
    const conduit::Node &edges_out(const std::string &f_name) const;

    std::map<std::string,Filter*> &filters();


    Workspace                       *m_workspace;
    conduit::Node                    m_edges;
    std::map<std::string,Filter*>    m_filters;
    int                              m_filter_count;

};


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


