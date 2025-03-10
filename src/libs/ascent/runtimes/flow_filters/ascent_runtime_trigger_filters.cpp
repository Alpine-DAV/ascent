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
#include <ascent_actions_utils.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#include <conduit_blueprint_mpi.hpp>
#endif

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
    bool res = check_string("condition",params, info, false);
    res &= check_string("callback",params, info, false);
    res &= check_string("actions_file",params, info, false);
    res &= check_list("actions_files",params, info, false);
    res &= check_list("actions",params, info, false);

    bool has_condition = params.has_child("condition");
    bool has_callback  = params.has_child("callback");
    bool has_actions   = params.has_child("actions");
    bool has_actions_file  = params.has_child("actions_file");
    bool has_actions_files = params.has_child("actions_files");

    if( has_condition && has_callback )
    {
      res = false;
      info["errors"].append() = "Both `condition` and `callback` are "
                                "present. Choose one or the other.";
    }

    if( ! (has_condition || has_callback) )
    {
      res = false;
      info["errors"].append() = "No `condition` or `callback` provided. "
                                "Choose please provide trigger `condition`"
                                " or `callback`";
    }

    if( has_actions_file && has_actions_files )
    {
      res = false;
      info["errors"].append() = "Both `actions_file` and `actions_files` are "
                                "present. Choose one or the other.";
    }

    if(has_actions && (has_actions_file || has_actions_files))
    {
      res = false;
      info["errors"].append() = "Both `actions` and `actions_file(s)` are "
                                "present. Choose one or the other.";
    }

    if(!has_actions && !(has_actions_file || has_actions_files))
    {
      res = false;
      info["errors"].append() = "No trigger actions provided. Please "
                                "specify either 'actions_file(s)' or "
                                "'actions'.";
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("condition");
    valid_paths.push_back("callback");
    valid_paths.push_back("actions_file");
    valid_paths.push_back("actions_files");
    valid_paths.push_back("actions");

    std::vector<std::string> ignore_paths;
    // don't go down the actions or actions_files path
    ignore_paths.push_back("actions");
    ignore_paths.push_back("actions_files");

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

    int mpi_comm_id = -1;

#ifdef ASCENT_MPI_ENABLED
      // TODO: Read from this Ascent instance's ops, vs static val
     mpi_comm_id = Workspace::default_mpi_comm();
#endif

    // params verify above will make sure that:
    //  actions, actions_file, and actions_files
    // are mutually exclusive options

    conduit::Node actions;

    std:vector<std::string> actions_files;

    if(params().has_child("actions_file"))
    {
        actions_files.push_back(params()["actions_file"].as_string());
    }
    else if(params().has_child("actions_files"))
    {
        NodeConstIterator itr = params()["actions_files"].children();
        while(itr.has_next())
        {
            actions_files.push_back(itr.next().as_string());
        }
    }
    else
    {
      actions = params()["actions"];
    }

    if(actions_files.size() > 0)
    {
        for(auto actions_file: actions_files)
        {
            Node loaded_actions;
            bool load_ok = load_actions_file(actions_file,
                                             mpi_comm_id,
                                             loaded_actions);
            if(!load_ok)
            {
                ASCENT_ERROR("Failed to load actions file: "
                             << actions_file);
            }
            
            if(!loaded_actions.dtype().is_list())
            {
                ASCENT_ERROR("Failed actions loaded from actions file: "
                             << actions_file << " are not a list");
            }

            NodeConstIterator itr = loaded_actions.children();
            while(itr.has_next())
            {
                const Node &curr = itr.next();
                actions.append().set(curr);
            }
        }
    }

    bool has_callback = params().has_path("callback");
    bool has_condition = params().has_path("condition");

    conduit::Node res;

    if(has_callback)
    {
      std::string callback_name = params()["callback"].as_string();
      res["value"] = ascent::execute_callback(callback_name);
      res["type"] = "bool";
    }
    else if(has_condition)
    {
      runtime::expressions::ExpressionEval eval(n_input.get());
      std::string expression = params()["condition"].as_string();
      res = eval.evaluate(expression);

      if(res["type"].as_string() != "bool")
      {
        ASCENT_ERROR("result of expression '"<<expression<<"' is not an bool");
      }
    }
    else
    {
        // NOTE: this is also handled in verify params
        ASCENT_ERROR("must provide either a `condition` or a `callback`");
    }

    bool fire = res["value"].to_uint8() != 0;

    if(fire)
    {
        Ascent ascent;
        Node ascent_opts;
#ifdef ASCENT_MPI_ENABLED
        ascent_opts["mpi_comm"] = mpi_comm_id;
#endif
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





