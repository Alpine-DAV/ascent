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
/// file: t_ascent_render_3d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent_expression_eval.hpp>
#include <ascent_hola.hpp>

#include <cmath>
#include <iostream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

index_t EXAMPLE_MESH_SIDE_DIM = 5;

//-----------------------------------------------------------------------------
TEST(ascent_expressions, derived_expressions)
{
  Node n;
  ascent::about(n);

  //
  // Create an example mesh.
  //
  Node data, verify_info;
  // conduit::blueprint::mesh::examples::braid("hexs",
  // conduit::blueprint::mesh::examples::braid("rectilinear",
  conduit::blueprint::mesh::examples::braid("uniform",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);
  // ascent normally adds this but we are doing an end around
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  //    +
  //  /   \
  //  1   + return_type: jitable
  //     / \
  // field  1
  //
  //       +
  //     /   \
  // field   + return_type: jitable
  //        / \
  //    field  1
  //
  //     max
  //      |
  //      +
  //     / \
  // field  1
  //
  // double braid = 1.;
  // double d = max((((double(2) + double(1)) / double(5.0000000000000000e-01))
  // + braid), double(0));
  // expr = "max((2.0 + 1) / 0.5 + field('braid'),0.0)";
  // expr = "test( foo = 1)";
  // expr = "sin(field('radial'))";

  // pass vec and see what happens
  // expr = "sin(field('braid')) * field('braid') * field('vel')";
  // expr = "field('braid') + 1 + field('braid') + 1";
  // eval.evaluate(expr);
  // expr = "binning_mesh(binning('braid','max', [axis('x', num_bins=10)]), "
  //        "name='binning')";
  // eval.evaluate(expr);
  // expr = "min(if field('binning') > .2 then abs(5 - "
  //        "topo('binning_topo').cell.x) else 1e18)";
  // res = eval.evaluate(expr);
  // res.print();

  // expr = "if field('braid') > 0 then field('braid') else 0";
  // eval.evaluate(expr);

  // expr = "topo('mesh').cell.x";
  // eval.evaluate(expr);

  // expr = "sin(field('braid'))";
  // eval.evaluate(expr);

  // expr = "(field('braid') - min(field('braid')).value) / "
  //        "(max(field('braid')).value - min(field('braid')).value)";
  // eval.evaluate(expr);

  // expr = "1 + field('braid') + 1";
  // eval.evaluate(expr);

  // expr = "max(field('braid') + 1)";
  // eval.evaluate(expr);

  const std::string output_path = prepare_output_dir();

  std::string output_file =
      conduit::utils::join_file_path(output_path, "fishtank_temp");

  conduit::Node actions;

  conduit::Node queries;
  expr = "binning_mesh(binning('temperature', 'sum', [axis('x', num_bins=50), "
         "axis('y', num_bins=50), axis('z', num_bins=50)], component='c0'), "
         "name='temp_binning')";
  queries["q1/params/expression"] = expr;
  queries["q1/params/name"] = "temp_binning";

  conduit::Node &add_queries = actions.append();
  add_queries["action"] = "add_queries";
  add_queries["queries"] = queries;

  conduit::Node extracts;
  extracts["e1/type"] = "relay";
  extracts["e1/params/path"] = output_file;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  // conduit::Node scenes;
  // scenes["s1/plots/p1/type"] = "pseudocolor";
  // scenes["s1/plots/p1/field"] = "temp_binning";
  // scenes["s1/renders/r1/image_prefix"] = output_file;
  // scenes["s1/renders/r1/render_bg"] = "false";
  //
  // conduit::Node &add_plots = actions.append();
  // add_plots["action"] = "add_scenes";
  // add_plots["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  ascent_opts["ascent_info"] = "verbose";
  ascent_opts["timings"] = "enabled";
  ascent_opts["runtime/type"] = "ascent";

  conduit::Node replay_data, replay_opts;
  replay_opts["root_file"] =
      "/Users/ibrahim5/datasets/fishtank/fishtank.cycle_000000.root";
  ascent::hola("relay/blueprint/mesh", replay_opts, replay_data);

  cout << actions.to_json();
  ascent.open(ascent_opts);
  ascent.publish(replay_data);
  ascent.execute(actions);
  ascent.close();
}

//-----------------------------------------------------------------------------

int
main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // allow override of the data size via the command line
  if(argc == 2)
  {
    EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
  }

  result = RUN_ALL_TESTS();
  return result;
}
