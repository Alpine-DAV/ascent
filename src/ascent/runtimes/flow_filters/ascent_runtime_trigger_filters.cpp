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

#include "ascent_runtime_trigger_filters.hpp"

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
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
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
BasicTrigger::BasicTrigger()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BasicTrigger::~BasicTrigger()
{
// empty
}

//-----------------------------------------------------------------------------
void
BasicTrigger::declare_interface(Node &i)
{
    i["type_name"]   = "basic_trigger";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
BasicTrigger::verify_params(const conduit::Node &params,
                            conduit::Node &info)
{
    info.reset();
    bool res = check_string("condition",params, info, true);

    res &= check_string("actions_file",params, info, false);
    if(params.has_path("actions"))
    {
      // basic actions node check
      if(!params["actions"].dtype().is_list())
      {
        res = false;
        info["errors"].append() = "trigger actions must be a node.";
      }
    }

    bool has_actions  = params.has_path("actions");
    bool has_actions_file  = params.has_path("actions_file");

    if(has_actions && has_actions_file)
    {
      res = false;
      info["errors"].append() = "Both actions and actions file are "
                                "present. Choose one or the other.";
    }

    if(!has_actions && !has_actions_file)
    {
      res = false;
      info["errors"].append() = "No trigger actions provided. Please "
                                "specify either 'actions_file' or "
                                "'actions'.";
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("condition");
    valid_paths.push_back("actions_file");
    valid_paths.push_back("actions");

    std::vector<std::string> ignore_paths;
    // don't go down the actions path
    ignore_paths.push_back("actions");

    std::string surprises = surprise_check(valid_paths, ignore_paths,params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}


//-----------------------------------------------------------------------------
void
BasicTrigger::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Trigger input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    std::shared_ptr<Node> n_input = data_object->as_low_order_bp();

    std::string expression = params()["condition"].as_string();

    bool use_actions_file = params().has_path("actions_file");

    std::string actions_file = "";
    conduit::Node actions;

    if(use_actions_file)
    {
      actions_file = params()["actions_file"].as_string();
    }
    else
    {
      actions = params()["actions"];
    }


    runtime::expressions::ExpressionEval eval(n_input.get());
    conduit::Node res = eval.evaluate(expression);

    if(res["type"].as_string() != "bool")
    {
      ASCENT_ERROR("result of expression '"<<expression<<"' is not an bool");
    }

    bool fire = res["value"].to_uint8() != 0;
    if(fire)
    {
      Ascent ascent;

      Node ascent_opts;
      ascent_opts["runtime/type"] = "ascent";
#ifdef ASCENT_MPI_ENABLED
      ascent_opts["mpi_comm"] = Workspace::default_mpi_comm();
#endif
      ascent_opts["actions_file"] = actions_file;
      ascent.open(ascent_opts);
      ascent.publish(*n_input);
      ascent.execute(actions);
      ascent.close();
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





