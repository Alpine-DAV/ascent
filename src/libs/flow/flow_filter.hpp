//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_filter.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_FILTER_HPP
#define FLOW_FILTER_HPP

#include <conduit.hpp>

#include <flow_exports.h>
#include <flow_config.h>

#include <flow_data.hpp>
#include <flow_registry.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{


class Workspace;
class Graph;

//-----------------------------------------------------------------------------
///
/// Filter Interface
///
///
/// Filters optionally provide:
///   - A set of named input ports
///   - output
///   - A set of default parameters
///
///  To create a new filter, create a new subclass of Filter and:
///
///  1) Implement declare_interface(), and adding the following entires
///  to a conduit Node.
///
///  void MyFilter::declare_interface(conduit::Node &i)
///  {
///    // unique filter name
///    i["type_name"]   = "my_filter";
///
///    // declare if this filter provides output
///    i["output_port"] = {"true" | "false"};
///
///    // declare the names of this filters input ports
///    // Provide a conduit list of strings with the names of the input ports
///    // or DataType::empty() if there are no input ports.
///    i["port_names"].append().set("in");
///
///    // Set any default parameters.
///    // default_params can be any conduit tree, params() will be
///    // inited with a *copy* of the default_params when the filter is
///    // added to the filter graph.
///    i["default_params"]["inc"].set((int)1);
///  }
///
///  2) Implement an execute() method:
///
///  void MyFilter::execute()
///  {
///     // If your filter has input ports, input data can be fetched by name
///     Node *in_0 = input<Node>("in");
///     // or index:
///     Node *in_0 = input<Node>(0);

///     // you can also check the type of an input using
///     // if(input("in").check_type<Node>) ...
///
///     // You can access filter parameters via params()
///     int val = params()["my_knob"].value();
///
///     // If your filter provides output, set your output data:
///     Node *my_result = new Node();
///     set_output<Node>(my_result);
///     // the registry manages result lifetimes.
///
///  }
///
///  TODO: talk about optional verify_params()
///
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class FLOW_API Filter
{
public:

    friend class Graph;
    friend class Workspace;

    virtual ~Filter();

    //-------------------------------------------------------------------------
    // subclasses need to implement these to define a filter:
    //-------------------------------------------------------------------------

    /// override and fill i with the info about the filter's interface
    virtual void          declare_interface(conduit::Node &i) = 0;

    /// override to imp filter's work
    virtual void          execute() = 0;


    /// optionally override to allow filter to verify custom params
    /// (used as a guard when a filter instance is created in a graph)
    virtual bool          verify_params(const conduit::Node &params,
                                        conduit::Node &info);

    //-------------------------------------------------------------------------
    // filter interface properties
    //-------------------------------------------------------------------------

    /// static method that checks if conduit node passed conforms to what
    /// is needed to declare a filter interface.
    /// (used as a guard when a filter type is registered)
    static bool           verify_interface(const conduit::Node &i,
                                           conduit::Node &info);

    /// helpers that provide access to the specific parts
    /// of the filter interface
    std::string           type_name()   const;
    const conduit::Node  &port_names()  const;
    bool                  output_port() const;

    const conduit::Node  &default_params() const;

    int                   number_of_input_ports() const;
    bool                  has_port(const std::string &name) const;
    std::string           port_index_to_name(int idx) const;

    std::string           name() const;
    std::string           detailed_name() const;

    //-------------------------------------------------------------------------
    // filter instance  properties
    //-------------------------------------------------------------------------

    // allows sub class to fetch the full interface def
    const conduit::Node   &interface() const;
    // allows sub class to fetch the params
    conduit::Node         &params();

    /// generic fetch of wrapped input data by port name
    Data &input(const std::string &port_name);
    /// generic fetch of wrapped input data by port index
    Data &input(int port_idx);

    /// templated fetch of wrapped input data by port name
    template <class T>
    T *input(const std::string &port_name)
    {
        return fetch_input(port_name)->value<T>();
    }

    /// templated fetch of wrapped input data by port index
    template <class T>
    T *input(int idx)
    {
        return fetch_input(idx)->value<T>();
    }


    /// generic set of wrapped output data
    void                   set_output(Data &data);

    /// templated set of wrapped output data
    template <class T>
    void set_output(T *data_ptr)
    {
        DataWrapper<T> data(data_ptr);
        set_output(data);
    }

    /// generic access to wrapped output data
    Data                  &output();

    /// templated access to  wrapped output data
    template <class T>
    T *output()
    {
        return output().value<T>();
    }

    /// access the filter's graph
    Graph                 &graph();

    /// connect helper
    /// equiv to:
    ///   graph().connect(f->name(),this->name(),port_name);
    void                  connect_input_port(const std::string &port_name,
                                             Filter *filter);

    /// connect helper
    /// equiv to:
    ///   graph().connect(f->name(),this->name(),idx);
    void                  connect_input_port(int idx,
                                             Filter *filter);


    /// create human understandable tree that describes the state
    /// of the filter
    void           info(conduit::Node &out) const;
    /// create json string from info
    std::string    to_json() const;
    /// create  yaml string from info
    std::string    to_yaml() const;
    /// print yaml version of info
    void           print() const;



protected:
    Filter();

private:

    conduit::Node          &interface();


    Data                   *fetch_input(const std::string &port_name);
    Data                   *fetch_input(int port_idx);
    // used by ws interface to imp data flow exec
    void                    set_input(const std::string &port_name,
                                      Data *data);

    void                    init(Graph *graph,
                                 const std::string &name,
                                 const conduit::Node &params);

    void                    reset_inputs_and_output();


    conduit::Node           &properties();
    const conduit::Node     &properties() const;


    Graph                        *m_graph;

    conduit::Node                 m_props;
    Data                         *m_out;
    std::map<std::string,Data*>   m_inputs;

};

//-----------------------------------------------------------------------------
typedef Filter *(*FilterFactoryMethod)(const std::string &filter_type_name);

//-----------------------------------------------------------------------------
template <class T>
Filter *CreateFilter(const std::string &filter_type_name)
{
    return new T;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


