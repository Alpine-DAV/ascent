//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_trigger_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_query_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_expression_eval.hpp>
#include <ascent_logging.hpp>
#include <ascent_data_object.hpp>
#include <ascent_runtime_param_check.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{


//-----------------------------------------------------------------------------
BasicQuery::BasicQuery()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BasicQuery::~BasicQuery()
{
// empty
}

//-----------------------------------------------------------------------------
void
BasicQuery::declare_interface(Node &i)
{
    i["type_name"]   = "basic_query";
    i["port_names"].append() = "in";
    // this is a dummy port that we use to enforce
    // a order of execution
    i["port_names"].append() = "dummy";
    // adding an output port to chain queries together
    // so they execute in order of declaration
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BasicQuery::verify_params(const conduit::Node &params,
                            conduit::Node &info)
{
    info.reset();
    bool res = check_string("expression",params, info, true);
    res &= check_string("name",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("expression");
    valid_paths.push_back("name");

    return res;
}


//-----------------------------------------------------------------------------
void
BasicQuery::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Query input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::string expression = params()["expression"].as_string();
    std::string name = params()["name"].as_string();
    conduit::Node actions;

    Node v_info;

    // The mere act of a query stores the results
    runtime::expressions::ExpressionEval eval(*data_object);
    conduit::Node res = eval.evaluate(expression, name);

    // we never actually use the output port
    // since we only use it to chain ordering
    conduit::Node *dummy =  new conduit::Node();
    set_output<conduit::Node>(dummy);
}

//-----------------------------------------------------------------------------
FilterQuery::FilterQuery()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
FilterQuery::~FilterQuery()
{
// empty
}

//-----------------------------------------------------------------------------
void
FilterQuery::declare_interface(Node &i)
{
    i["type_name"]   = "expression";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FilterQuery::verify_params(const conduit::Node &params,
                            conduit::Node &info)
{
    info.reset();
    bool res = check_string("expression",params, info, true);
    res &= check_string("name",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("expression");
    valid_paths.push_back("name");

    return res;
}


//-----------------------------------------------------------------------------
void
FilterQuery::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Query input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);

    std::string expression = params()["expression"].as_string();
    std::string name = params()["name"].as_string();
    conduit::Node actions;

    Node v_info;

    // The mere act of a query stores the results
    //runtime::expressions::ExpressionEval eval(n_input.get());
    runtime::expressions::ExpressionEval eval(*data_object);
    conduit::Node res = eval.evaluate(expression, name);

    // if the end result is a derived field the for sure we want to make
    // its available.
    bool derived = false;
    if(res.has_path("type"))
    {
      if(res["type"].as_string() == "field")
      {
        derived = true;
      }
    }

    // Since queries might add new fields, the blueprint needs to become the source
    if(derived && (data_object->source() != DataObject::Source::LOW_BP))
    {
      // for now always copy the bp if its not the original data source
      // There is one main reasons for this:
      //   the data will likely be passed to the vtkh ghost stripper, which could create
      //   a new data sets with memory owned by vtkm. Since conduit can't take ownership of
      //   that memory, this data could could go out of scope and that would be bad. To ensure
      //   that it does not go out of scope
      //   TODO: We could be smarter than this. For example, we could provide a way to map a
      //   new field, if created, back on to the original source (e.g., vtkm)
      conduit::Node *new_source = new conduit::Node(*eval.data_object().as_low_order_bp());
      DataObject *new_do = new DataObject(new_source);
      set_output<DataObject>(new_do);
    }
    else
    {
      set_output<DataObject>(data_object);
    }

}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





