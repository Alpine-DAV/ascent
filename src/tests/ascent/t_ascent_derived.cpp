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

index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_expressions, derived_expressions)
{
  Node n;
  ascent::about(n);

  //
  // Create an example mesh.
  //
  Node data, verify_info;
  // conduit::blueprint::mesh::examples::basic("polyhedra",
  // conduit::blueprint::mesh::examples::basic("polygons",
  // conduit::blueprint::mesh::examples::braid("tris",
  // conduit::blueprint::mesh::examples::braid("quads",
  // conduit::blueprint::mesh::examples::braid("tets",
  // conduit::blueprint::mesh::examples::braid("hexs",
  conduit::blueprint::mesh::examples::braid(
      "uniform",
      // conduit::blueprint::mesh::examples::braid("rectilinear",
      // conduit::blueprint::mesh::examples::braid("structured",
      EXAMPLE_MESH_SIDE_DIM,
      EXAMPLE_MESH_SIDE_DIM,
      EXAMPLE_MESH_SIDE_DIM,
      // 0,
      data);
  // ascent normally adds this but we are doing an end around
  data["state/domain_id"] = 0;
  Node multi_dom;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  // expr = "1 + field('braid') + 1";
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
  //      + jitable -> field
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
  // expr = "binning = binning('', 'cdf', [axis('braid', num_bins=10)])\n"
  //        "if(binning_value(binning) < rand()) then 0 else field('braid')";
  // eval.evaluate(expr);
  // expr = "min(if field('binning') > .2 then abs(5 - "
  //        "topo('binning_topo').cell.x) else 1e18)";
  // res = eval.evaluate(expr);
  // res.print();

  // expr = "if field('braid') > 0 then field('braid') else if field('braid') >
  // -1 then -1 else 0"; eval.evaluate(expr);

  // expr = "topo('mesh').cell.x";
  // eval.evaluate(expr);

  // expr = "topo('mesh').cell.area";
  // eval.evaluate(expr);

  // expr = "topo('mesh').cell.volume";
  // eval.evaluate(expr);

  // expr = "magnitude(gradient(field('braid') + 1))";
  // eval.evaluate(expr);

  // expr = "sin(field('braid'))";
  // eval.evaluate(expr);

  // expr = "(field('braid') - min(field('braid')).value) / "
  //        "(max(field('braid')).value - min(field('braid')).value)";
  // eval.evaluate(expr);

  // expr = "1 + field('braid') + 1";
  // eval.evaluate(expr);

  // expr = "f1 = 1 + field('braid') \n"
  //        "f1 + 1";
  // eval.evaluate(expr);

  // expr = "field('field') * topo('mesh').cell.volume";
  // eval.evaluate(expr, "mass");

  // expr = "recenter(field('braid') + 1)";
  // eval.evaluate(expr);

  // conduit::Node m, info;
  // std::cout << "interleaved" << std::endl;
  // conduit::blueprint::mcarray::examples::xyz("interleaved", 10, m);
  // m.info(info);
  // info.print();
  // std::cout << "is_obj" << (m.child(0).dtype().id() == DataType::OBJECT_ID)
  // << std::endl; std::cout << "dtype().is_compact: " << m.dtype().is_compact()
  // << std::endl; std::cout << "dtype().stride: " <<
  // m.child(0).dtype().stride() << std::endl; std::cout <<
  // "dtype().total_strided_bytes: " <<  m.total_strided_bytes() << std::endl;
  // std::cout << "dtype().strided_bytes: " <<
  // m.child(1).dtype().strided_bytes() << std::endl; std::cout <<
  // "dtype().spanned_bytes: " <<  m.child(1).dtype().spanned_bytes() <<
  // std::endl; std::cout << "child(0).data_ptr: " <<  m.child(0).data_ptr() <<
  // std::endl; std::cout << "child(1).data_ptr: " <<  m.child(1).data_ptr() <<
  // std::endl; std::cout << "child(2).data_ptr: " <<  m.child(2).data_ptr() <<
  // std::endl; std::cout << "number_of_children: " <<  m.number_of_children()
  // << std::endl; std::cout << "is_contiguous: " <<  m.is_contiguous() <<
  // std::endl; std::cout << "is_compact: " <<  m.is_compact() << std::endl;
  // m.print_detailed();
  // std::cout << "separate" << std::endl;
  // conduit::blueprint::mcarray::examples::xyz("separate", 10, m);
  // m.info(info);
  // info.print();
  // std::cout << "dtype().total_strided_bytes: " <<  m.total_strided_bytes() <<
  // std::endl; std::cout << "dtype().is_compact: " <<  m.dtype().is_compact()
  // << std::endl; std::cout << "dtype().stride: " <<
  // m.child(0).dtype().stride() << std::endl; std::cout <<
  // "dtype().strided_bytes: " <<  m.child(0).dtype().strided_bytes() <<
  // std::endl; std::cout << "dtype().spanned_bytes: " <<
  // m.child(0).dtype().spanned_bytes() << std::endl; std::cout <<
  // "child(0).data_ptr: " <<  m.child(0).data_ptr() << std::endl; std::cout <<
  // "child(1).data_ptr: " <<  m.child(1).data_ptr() << std::endl; std::cout <<
  // "child(2).data_ptr: " <<  m.child(2).data_ptr() << std::endl; std::cout <<
  // "number_of_children: " <<  m.number_of_children() << std::endl; std::cout
  // << "is_contiguous: " <<  m.is_contiguous() << std::endl; std::cout <<
  // "is_compact: " <<  m.is_compact() << std::endl; m.print_detailed();
  // std::cout << "contiguous" << std::endl;
  // conduit::blueprint::mcarray::examples::xyz("contiguous", 10, m);
  // m.info(info);
  // info.print();
  // std::cout << "dtype().total_strided_bytes: " <<  m.total_strided_bytes() <<
  // std::endl; std::cout << "dtype().is_compact: " <<  m.dtype().is_compact()
  // << std::endl; std::cout << "dtype().stride: " <<
  // m.child(0).dtype().stride() << std::endl; std::cout <<
  // "dtype().strided_bytes: " <<  m.child(0).dtype().strided_bytes() <<
  // std::endl; std::cout << "dtype().spanned_bytes: " <<
  // m.child(0).dtype().spanned_bytes() << std::endl; std::cout <<
  // "child(0).data_ptr: " <<  m.child(0).data_ptr() << std::endl; std::cout <<
  // "child(1).data_ptr: " <<  m.child(1).data_ptr() << std::endl; std::cout <<
  // "child(2).data_ptr: " <<  m.child(2).data_ptr() << std::endl; std::cout <<
  // "number_of_children: " <<  m.number_of_children() << std::endl; std::cout
  // << "is_contiguous: " <<  m.is_contiguous() << std::endl; std::cout <<
  // "is_compact: " <<  m.is_compact() << std::endl; m.print_detailed();
}

/*
TEST(ascent_expressions, derived_temperature)
{
  conduit::Node replay_data, replay_opts;
  // replay_opts["root_file"] =
  // "/Users/ibrahim5/datasets/fishtank/fishtank.cycle_000000.root";
  replay_opts["root_file"] =
      "/Users/ibrahim5/datasets/sharknato/sharknato_7221.cycle_000000.root";
  std::cout << "importing..." << std::endl;
  ascent::hola("relay/blueprint/mesh", replay_opts, replay_data);
  std::cout << "done importing..." << std::endl;

  // conduit::Node data, multi_dom;
  // conduit::blueprint::mesh::examples::braid("structured",
  //                                           EXAMPLE_MESH_SIDE_DIM,
  //                                           EXAMPLE_MESH_SIDE_DIM,
  //                                           0,
  //                                           data);
  // data["state/domain_id"] = 0;
  // blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&replay_data);
  // runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  // expr = "gradient(field('temperature', 'c0'))";
  // eval.evaluate(expr, "temp_gradient");
  // expr = "gradient(field('uinterp', 'c0'))";
  // eval.evaluate(expr, "uinterp_gradient");
  // expr = "gradient(field('braid'))";
  // eval.evaluate(expr, "braid_grad");
  // expr = "vorticity(field('uinterp', 'c0'))";
  // eval.evaluate(expr, "uinterp_vorticity");

  // expr = "vorticity(vector(field('uinterp', 'c0'), field('vinterp', 'c0'),
  // field('winterp', 'c0')))"; eval.evaluate(expr, "velocity_vorticity");

  expr = "du = gradient(field('uinterp'))\n"
         "dv = gradient(field('vinterp'))\n"
         "dw = gradient(field('winterp'))\n"
         "w_x = dw.y - dv.z\n"
         "w_y = du.z - dw.x\n"
         "w_z = dv.x - du.y\n"
         "magnitude(vector(w_x, w_y, w_z))";
  eval.evaluate(expr, "velocity_vorticity");

  // expr = "vector(field('uinterp', 'c0'), field('vinterp', 'c0'),
  // field('winterp', 'c0'))"; eval.evaluate(expr, "velocity");

  // expr = "vorticity(field('velocity'))";
  // eval.evaluate(expr, "velocity_vorticity");

  const std::string output_path = prepare_output_dir();

  std::string output_file_image =
      conduit::utils::join_file_path(output_path, "vorticity_image");
  std::string output_file_hdf5 =
      conduit::utils::join_file_path(output_path, "vorticity");

  conduit::Node actions;

  conduit::Node extracts;
  extracts["e1/type"] = "relay";
  extracts["e1/params/path"] = output_file_hdf5;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "volume";
  scenes["s1/plots/p1/field"] = "velocity_vorticity";
  // scenes["s1/plots/p1/min_value"] = 500;
  scenes["s1/plots/p1/color_table/name"] = "rainbow desaturated";
  conduit::Node &c1 = scenes["s1/plots/p1/color_table/control_points"].append();
  c1["type"] = "alpha";
  c1["position"] = 0;
  c1["alpha"] = 0;
  conduit::Node &c2 = scenes["s1/plots/p1/color_table/control_points"].append();
  c2["type"] = "alpha";
  c2["position"] = .05;
  c2["alpha"] = 0;
  conduit::Node &c3 = scenes["s1/plots/p1/color_table/control_points"].append();
  c3["type"] = "alpha";
  c3["position"] = 1;
  c3["alpha"] = 1;
  scenes["s1/renders/r1/image_prefix"] = output_file_image;
  // scenes["s1/renders/r1/camera/azimuth"] = 35.0;
  scenes["s1/renders/r1/camera/zoom"] = 1.5;
  double look_at[3] = {-1, 0, 0};
  double up[3] = {0, 0, 1};
  double position[3] = {-2, -2, 3};
  scenes["s1/renders/r1/camera/look_at"].set(look_at, 3);
  scenes["s1/renders/r1/camera/up"].set(up, 3);
  // scenes["s1/renders/r1/camera/position"].set(position, 3);
  // scenes["s1/renders/r1/camera/fov"] = 30.0;
  // scenes["s1/renders/r1/camera/xpan"] = 0;
  // scenes["s1/renders/r1/camera/ypan"] = 0;
  // scenes["s1/renders/r1/camera/zoom"] = .1;
  // scenes["s1/renders/r1/camera/near_plane"] = -18.26;
  // scenes["s1/renders/r1/camera/far_plane"] = 18.26;

  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  ascent_opts["ascent_info"] = "verbose";
  ascent_opts["timings"] = "enabled";
  ascent_opts["runtime/type"] = "ascent";

  // multi_dom.print();
  ascent.open(ascent_opts);
  ascent.publish(replay_data);
  // ascent.publish(multi_dom);
  ascent.execute(actions);
  ascent.close();
}
*/

TEST(ascent_expressions, braid_sample)
{
  conduit::Node data, multi_dom;
  conduit::blueprint::mesh::examples::braid("structured",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);
  data["state/domain_id"] = 0;
  blueprint::mesh::to_multi_domain(data, multi_dom);

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&multi_dom);

  conduit::Node res;
  std::string expr;

  expr = "binning = binning('cnt', 'cdf', [axis(field('braid'), num_bins=10)])\n"
         "if(binning_value(binning) < rand()) then 0 else field('braid')";
  eval.evaluate(expr, "braid_sample");

  const std::string output_path = prepare_output_dir();

  std::string output_file_image =
      conduit::utils::join_file_path(output_path, "braid_sample_image");
  std::string output_file_braid =
      conduit::utils::join_file_path(output_path, "braid_original");
  std::string output_file_hdf5 =
      conduit::utils::join_file_path(output_path, "braid_sample");

  conduit::Node actions;

  conduit::Node extracts;
  extracts["e1/type"] = "relay";
  extracts["e1/params/path"] = output_file_hdf5;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "volume";
  scenes["s1/plots/p1/field"] = "braid_sample";
  scenes["s1/plots/p1/color_table/name"] = "rainbow desaturated";
  scenes["s1/renders/r1/image_prefix"] = output_file_image;
  scenes["s1/renders/r1/camera/azimuth"] = 35.0;

  scenes["s2/plots/p1/type"] = "volume";
  scenes["s2/plots/p1/field"] = "braid";
  scenes["s2/plots/p1/color_table/name"] = "rainbow desaturated";
  scenes["s2/renders/r1/image_prefix"] = output_file_braid;
  scenes["s2/renders/r1/camera/azimuth"] = 35.0;

  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  //
  // Run Ascent
  //

  Ascent ascent;

  Node ascent_opts;
  ascent_opts["ascent_info"] = "verbose";
  ascent_opts["timings"] = "enabled";
  ascent_opts["runtime/type"] = "ascent";

  ascent.open(ascent_opts);
  ascent.publish(multi_dom);
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
