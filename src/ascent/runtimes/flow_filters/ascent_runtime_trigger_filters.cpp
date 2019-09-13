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
    res &= check_string("actions_file",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("condition");
    valid_paths.push_back("actions_file");

    return res;
}


//-----------------------------------------------------------------------------
void
BasicTrigger::execute()
{
    if(!input(0).check_type<Node>())
    {
        ASCENT_ERROR("Trigger input must be a conduit node");
    }

    std::string expression = params()["condition"].as_string();
    std::string actions_file = params()["actions_file"].as_string();
    conduit::Node actions;

    Node v_info;
    Node *n_input = input<Node>(0);

    runtime::expressions::ExpressionEval eval(n_input);
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





