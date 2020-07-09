//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
/// file: ascent_runtime_trigger_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_newtrigger_filters.hpp"

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
NewTrigger::NewTrigger()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
NewTrigger::~NewTrigger()
{
// empty
}

//-----------------------------------------------------------------------------
void
NewTrigger::declare_interface(Node &i)
{
    i["type_name"]   = "trigger";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
NewTrigger::verify_params(const conduit::Node &params,
                            conduit::Node &info)
{
    info.reset();
    bool res = true;
/*    bool res = check_string("condition",params, info, true);

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
    */

    return res;
}


//-----------------------------------------------------------------------------
void
NewTrigger::execute()
{
    std::cout << "Hey Yuya, you're now here!" << std::endl;
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Trigger input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    /*
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
    */
    set_output<DataObject>(input<DataObject>(0));
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





