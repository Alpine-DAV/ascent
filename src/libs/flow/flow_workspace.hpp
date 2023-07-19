//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_workspace.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_WORKSPACE_HPP
#define FLOW_WORKSPACE_HPP

#include <conduit.hpp>

#include <flow_exports.h>
#include <flow_config.h>
#include <flow_data.hpp>
#include <flow_registry.hpp>
#include <flow_graph.hpp>
#include <sstream>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
///
/// Workspace
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class FLOW_API Workspace
{
public:

    friend class Graph;

   // ------------------------------------------------------------------------
   /// Workspace instance methods
   // ------------------------------------------------------------------------

    Workspace();
   ~Workspace();

    /// access the filter graph
    Graph           &graph();
    /// const access to the filter graph
    const Graph     &graph() const;

    /// access the registry
    Registry        &registry();
    /// const access to the registry
    const Registry  &registry() const;

    /// compute and return the graph traverals
    void             traversals(conduit::Node &out);

    /// execute the filter graph.
    void             execute();

    /// reset the registry and graph
    void             reset();

    /// create human understandable tree that describes the state
    /// of the workspace
    void           info(conduit::Node &out) const;
    /// create json string from info
    std::string    to_json() const;
    /// create yaml string from info
    std::string    to_yaml() const;
    /// print yaml version of info
    void           print() const;

    /// resets state used to capture timing events
    void           reset_timing_info();
    /// return a string of recorded timing events
    std::string    timing_info() const;

    // ------------------------------------------------------------------------
    /// Interface to set and obtain the MPI communicator.
    ///
    /// We use an integer handle from MPI_Comm_c2f to avoid
    /// a header dependency of mpi just for the handle.
    ///
    // ------------------------------------------------------------------------
    void static set_default_mpi_comm(int mpi_comm_id);
    int  static default_mpi_comm();

    // ------------------------------------------------------------------------
    /// filter factory interface
    // ------------------------------------------------------------------------

    /// register a new type
    static void register_filter_type(FilterFactoryMethod fr);

    /// register a new type
    static void register_filter_type(const std::string &filter_type_name,
                                     FilterFactoryMethod fr);

    /// check if type with given name is registered
    static bool supports_filter_type(const std::string &filter_type_name);

    // /// check if type with given factory is registered
    static bool supports_filter_type(FilterFactoryMethod fr);

    /// remove type with given name if registered
    static void remove_filter_type(const std::string &filter_type_name);

    // /// returns the filter name for a registered filter
    static std::string filter_type_name(FilterFactoryMethod fr);

    /// remove all registered types
    static void clear_supported_filter_types();

    /// helper to for registering a filter type that does not provide its own
    /// FilterFactoryMethod
    template <class T>
    static void register_filter_type()
    {
        register_filter_type(&CreateFilter<T>);
    }

    /// helper for checkeding if a filter type is registered
    template <class T>
    static bool supports_filter_type()
    {
        return supports_filter_type(&CreateFilter<T>);
    }


    /// helper for checkeding if a filter type is registered
    template <class T>
    static std::string filter_type_name()
    {
        return filter_type_name(&CreateFilter<T>);
    }

    void enable_timings(bool enabled);

private:

    static Filter *create_filter(const std::string &filter_type);

    static int  m_default_mpi_comm;

    class ExecutionPlan;
    class FilterFactory;

    Graph             m_graph;
    Registry          m_registry;
    std::stringstream m_timing_info;
    bool              m_enable_timings;

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


