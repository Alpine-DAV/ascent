// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/threshold.hpp>
#include <string>

int EXAMPLE_MESH_SIDE_DIM = 20;

#define DEBUG_TEST
//#define GENERATE_BASELINES

//-----------------------------------------------------------------------------
#ifdef _WIN32
const std::string sep("\\");
#else
const std::string sep("/");
#endif

//-----------------------------------------------------------------------------
std::string
baseline_dir()
{
    return dray_baselines_dir() + sep + "threshold";
}

//-----------------------------------------------------------------------------
void
dray_collection_to_blueprint(dray::Collection &c, conduit::Node &n)
{
  int i = 0;
  for(auto it = c.domains().begin();
      it != c.domains().end(); it++, i++)
  {
      std::stringstream s;
      s << "domain" << i;
      conduit::Node dnode;
      try
      {
          it->to_node(dnode);
          // Now, take the dray conduit node and convert to blueprint so
          // we can actually look at it in VisIt.
          std::string path(s.str());
          conduit::Node &bnode = n[path];
          dray::BlueprintLowOrder::to_blueprint(dnode, bnode);
      }
      catch(std::exception &e)
      {
          std::cerr << "EXCEPTION:" << e.what() << std::endl;
      }
  }
}

//-----------------------------------------------------------------------------
void
make_baseline(const std::string &testname, const conduit::Node &tdata)
{
    std::string filename(baseline_dir() + sep + testname + ".yaml");
    std::string protocol("yaml");
    conduit::relay::io::save(tdata, filename, protocol);
}

//-----------------------------------------------------------------------------
void
convert_float64_to_float32(conduit::Node &input)
{
    // Traverse the Conduit data tree and transmute float64 things to float32
    // We have to do this because dray is making float32 and conduit reading back
    // from disk with a user-readable format makes float64 by default. Then the
    // diff function considers them not equal.
    for(conduit::index_t i = 0; i < input.number_of_children(); i++)
    {
        conduit::Node &n = input[i];
        if(n.dtype().is_float64())
        {
            auto arr = n.as_float64_array();
            std::vector<float> tmp;
            tmp.reserve(arr.number_of_elements());
            for(conduit::index_t elem = 0; elem < arr.number_of_elements(); elem++)
                tmp.push_back(static_cast<float>(arr[elem]));
            n.set(tmp);
        }
        else
        {
            convert_float64_to_float32(n);
        }
    }
}

//-----------------------------------------------------------------------------
void
compare_baseline(const std::string &testname, const conduit::Node &tdata)
{
    std::string filename(baseline_dir() + sep + testname + ".yaml");
    conduit::Node baseline, info;
    conduit::relay::io::load(filename, "yaml", baseline);
    convert_float64_to_float32(baseline);
    bool different = baseline.diff(tdata, info, 1.e-4, true);
    if(different)
    {
        std::cout << "Difference for " << testname << std::endl;
        info.print();
    }
    EXPECT_EQ(different, false);
}

//-----------------------------------------------------------------------------
void
handle_test(const std::string &testname, dray::Collection &tfdataset)
{
  // We save the Blueprint-compatible dataset as the baseline.
  conduit::Node tfdata;
  dray_collection_to_blueprint(tfdataset, tfdata);
#ifdef GENERATE_BASELINES
  make_baseline(testname, tfdata);
  #ifdef DEBUG_TEST
  //tfdata.print();
  dray::BlueprintReader::save_blueprint(testname, tfdata);
  #endif
#else
  compare_baseline(testname, tfdata);
#endif
}

#ifdef DEBUG_TEST
//-----------------------------------------------------------------------------
void
blueprint_plugin_error_handler(const std::string &msg,
                               const std::string &file,
                               int line)
{
    std::cout << "[ERROR]"
               << "File:"    << file << std::endl
               << "Line:"    << line << std::endl
               << "Message:" << msg  << std::endl;
    while(1); // hang!
}
#endif

//-----------------------------------------------------------------------------
TEST (dray_mesh_threshold, structured)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);
  dray::DataSet domain = dray::BlueprintLowOrder::import(data);
  dray::Collection dataset;
  dataset.add_domain(domain);
#ifdef DEBUG_TEST
  // Write the original data.
  conduit::Node inputdata;
  dray_collection_to_blueprint(dataset, inputdata);
  //inputdata.print();
  dray::BlueprintReader::save_blueprint("structured", inputdata);
#endif
  conduit::Node tfdata;

  // point-centered, any in range.
  dray::Threshold tf;
  tf.set_lower_threshold(-10.);
  tf.set_upper_threshold(0.);
  tf.set_field("braid");
  tf.set_all_in_range(false);
  auto tfdataset = tf.execute(dataset);
  handle_test("structured_braid_any", tfdataset);

  // point-centered, all in range.
  tf.set_all_in_range(true);
  tfdataset = tf.execute(dataset);
  handle_test("structured_braid_all", tfdataset);

  // cell-centered
  tf.set_field("radial");
  tf.set_lower_threshold(0.);
  tf.set_upper_threshold(90.);
  tfdataset = tf.execute(dataset);
  handle_test("structured_radial", tfdataset);
}
