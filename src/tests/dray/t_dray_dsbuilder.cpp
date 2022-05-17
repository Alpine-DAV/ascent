// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/utils/dataset_builder.hpp>
#include <conduit.hpp>
#include <conduit_relay.hpp>

TEST (dray_dsbuilder, dray_dsbuilder_simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "dsbuilder_simple_bp");

  std::vector<std::string> mesh_names = {"reference", "world"};

  std::vector<std::string> vfield_names = {"red", "blue"};
  std::vector<std::string> efield_names = {"green"};

  std::vector<std::string> vvectr_names = {"v_vector"};
  std::vector<std::string> evectr_names = {"e_vector"};

  dray::DataSetBuilder dsbuilder(dray::DataSetBuilder::Hex,
                                 mesh_names,
                                 vfield_names,
                                 efield_names,
                                 vvectr_names,
                                 evectr_names);
  dsbuilder.resize_num_buffers(3);

  dray::HexRecord hex_record = dsbuilder.new_empty_hex_record();

  hex_record.coord_data("reference", {{ {{0,0,0}},
                                        {{1,0,0}},
                                        {{0,1,0}},
                                        {{1,1,0}},
                                        {{0,0,1}},
                                        {{1,0,1}},
                                        {{0,1,1}},
                                        {{1,1,1}} }});

  hex_record.coord_data("world", {{ {{0,0,0}},
                                    {{5,0,0}},
                                    {{0,5,0}},
                                    {{5,5,0}},
                                    {{0,0,5}},
                                    {{5,0,5}},
                                    {{0,5,5}},
                                    {{5,5,5}} }});

  hex_record.scalar_vert_data("red",  {{ {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}} }});
  hex_record.scalar_vert_data("blue", {{ {{0}}, {{1}}, {{0}}, {{1}}, {{10}}, {{12}}, {{14}}, {{16}} }});
  hex_record.scalar_elem_data("green", {{ {{101}} }});

  hex_record.vector_vert_data("v_vector", {{ {{-1,-1,-1}},
                                         {{ 1,-1,-1}},
                                         {{-1, 1,-1}},
                                         {{ 1, 1,-1}},
                                         {{-1,-1, 1}},
                                         {{ 1,-1, 1}},
                                         {{-1, 1, 1}},
                                         {{ 1, 1, 1}} }});

  hex_record.vector_elem_data("e_vector", {{ {{0, 0, 1}} }});

  hex_record.birthtime(0);
  dsbuilder.add_hex_record(2, hex_record);

  hex_record.reuse_all();
  hex_record.coord_data("reference", {{ {{0,0,0}},
                                        {{.5,0,0}},
                                        {{0,.5,0}},
                                        {{.5,.5,0}},
                                        {{0,0,.5}},
                                        {{.5,0,.5}},
                                        {{0,.5,.5}},
                                        {{.5,.5,.5}} }});

  hex_record.birthtime(1);
  hex_record.immortal(true);
  dsbuilder.add_hex_record(0, hex_record);

  hex_record.reuse_all();
  hex_record.coord_data("reference", {{ {{.5,.5,.5}},
                                        {{1.,.5,.5}},
                                        {{.5,1.,.5}},
                                        {{1.,1.,.5}},
                                        {{.5,.5,1.}},
                                        {{1.,.5,1.}},
                                        {{.5,1.,1.}},
                                        {{1.,1.,1.}} }});

  hex_record.birthtime(2);
  hex_record.immortal(true);
  dsbuilder.add_hex_record(1, hex_record);

  hex_record.reuse_all();
  hex_record.coord_data("reference", {{ {{.5,0.,0.}},
                                        {{1.,0.,0.}},
                                        {{.5,.5,0.}},
                                        {{1.,.5,0.}},
                                        {{.5,0.,.5}},
                                        {{1.,0.,.5}},
                                        {{.5,.5,.5}},
                                        {{1.,.5,.5}} }});

  hex_record.birthtime(2);                      // Same timestep,
  hex_record.immortal(true);                    // but add to
  dsbuilder.add_hex_record(0, hex_record);      // a different buffer.

  dsbuilder.clear_buffer(1);                    // Then clear the original buffer.
  dsbuilder.flush_and_close_all_buffers();

  conduit::Node mesh;
  /// const std::string extension = ".blueprint_root_hdf5";
  const std::string extension = ".blueprint_root";  // visit sees time series if use json format.
  for (int32 cycle = 0; cycle < dsbuilder.num_timesteps(); ++cycle)
  {
    char cycle_suffix[8] = "_000000";
    snprintf(cycle_suffix, 8, "_%06d", cycle);

    const int32 n_sel_elems = dsbuilder.to_blueprint(mesh, cycle);
    if (n_sel_elems > 0)
      conduit::relay::io::blueprint::save_mesh(mesh, output_file + std::string(cycle_suffix) + extension);
  }
}
