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
#include <dray/filters/clipfield.hpp>
#include <dray/filters/clip.hpp>
#include <string>

int EXAMPLE_MESH_SIDE_DIM = 20;

#define DEBUG_TEST
#define GENERATE_BASELINES

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
    return dray_baselines_dir() + sep + "clipfield";
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
void
clip_3d(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::ClipField clip;
  clip.set_clip_value(-4.8);
  clip.set_field("braid"); // mesh_topology/braid

  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_") + name, output);

  // Filter again, inverting the selection.
  clip.set_invert_clip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_sphere(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::Clip clip;
  dray::Float center[]={0., 0., 0.}, radius = 5.;
  clip.SetSphereClip(center, radius);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_sphere_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_sphere_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_1_plane(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::Clip clip;
  dray::Float origin[]={0., 0., 0.}, normal[] = {2., 0., 1.}; // it'll normalize
  clip.SetPlaneClip(origin, normal);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_1_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_1_plane_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_2_plane(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::Clip clip;
  dray::Float origin1[]={0., 0., 0.}, normal1[] = {2., 0., 1.}; // it'll normalize
  dray::Float origin2[]={-2., 0., 0.}, normal2[] = {1., 0., 0.};
  clip.Set2PlaneClip(origin1, normal1, origin2, normal2);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_1_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_1_plane_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_3_plane(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::Clip clip;
  dray::Float origin1[]={0., 0., 0.}, normal1[] = {2., 0., 1.}; // it'll normalize
  dray::Float origin2[]={-2., 0., 0.}, normal2[] = {1., 0., 0.};
  dray::Float origin3[]={0., -1., 0.}, normal3[] = {0., 1., 0.};

  clip.Set3PlaneClip(origin1, normal1, origin2, normal2, origin3, normal3);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_1_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_1_plane_inv_") + name, output2);
}

#if 0
//-----------------------------------------------------------------------------
TEST (dray_clipfield_threshold, unstructured_hex)
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

  clip_3d(data, "explicit_hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip_sphere, unstructured_hex)
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

  clip_sphere(data, "explicit_hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip_1_plane, unstructured_hex)
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

  clip_1_plane(data, "explicit_hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip_2_plane, unstructured_hex)
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

  clip_2_plane(data, "explicit_hexs");
}
#endif

//-----------------------------------------------------------------------------
TEST (dray_clip_3_plane, unstructured_hex)
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

  clip_3_plane(data, "explicit_hexs");
}
