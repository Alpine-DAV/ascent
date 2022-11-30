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

int EXAMPLE_MESH_SIDE_DIM = 15;
int EXAMPLE_MESH_SIDE_DIM_SM = 7;

//#define DEBUG_TEST
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

          bnode["state/domain_id"] = it->domain_id();
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
limit_precision(conduit::Node &input)
{
    const int p = 1000;
    for(conduit::index_t i = 0; i < input.number_of_children(); i++)
    {
        conduit::Node &n = input[i];
        if(n.dtype().is_float64())
        {
            auto arr = n.as_float64_array();
            std::vector<double> tmp;
            tmp.reserve(arr.number_of_elements());
            for(conduit::index_t elem = 0; elem < arr.number_of_elements(); elem++)
            {
                double val = arr[elem];
                val = static_cast<double>(static_cast<int>(val * p) % p) / static_cast<double>(p);
                tmp.push_back(val);
            }
            n.set(tmp);
        }
        else if(n.dtype().is_float32())
        {
            auto arr = n.as_float32_array();
            std::vector<float> tmp;
            tmp.reserve(arr.number_of_elements());
            for(conduit::index_t elem = 0; elem < arr.number_of_elements(); elem++)
            {
                float val = arr[elem];
                val = static_cast<float>(static_cast<int>(val * p) % p) / static_cast<float>(p);
                tmp.push_back(val);
            }
            n.set(tmp);
        }
        else
        {
            limit_precision(n);
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
    bool different = baseline.diff(tdata, info, 1.e-2, true);
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
//  limit_precision(tfdata);
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

//-----------------------------------------------------------------------------
// Chris Laganella: I had to rebaseline a couple tests in this file and wanted
//  to double check the vector fields were being treated properly - so I grabbed
//  this from the extract_slice tests.
static conduit::Node
make_simple_field(int constant, int nx, int ny, int nz, int nc=1)
{
  std::vector<std::vector<float>> data;
  data.resize(nc);

  for(int i = 0; i < nz; i++)
  {
    for(int j = 0; j < ny*nx; j++)
    {
      for(int c = 0; c < nc; c++)
      {
        if(c % 2 == 0)
        {
          data[c].push_back(float(constant * (i)));
        }
        else
        {
          data[c].push_back(-float(constant * (i)));
        }
      }
    }
  }

#ifndef NDEBUG
  std::cout << "input:";
  for(int i = 0; i < data[0].size(); i++)
  {
    std::cout << " (";
    for(int c = 0; c < nc; c++)
    {
      std::cout << data[c][i] << (c < nc-1 ? "," : ")");
    }
  }
  std::cout << std::endl;
#endif

  conduit::Node n_field;
  if(nc == 1)
  {
    // Scalar
    n_field["association"].set("vertex");
    n_field["type"].set("scalar");
    n_field["topology"].set("mesh");
    n_field["values"].set(data[0]);
  }
  else
  {
    // Vector
    n_field["association"].set("vertex");
    n_field["type"].set("vector");
    n_field["topology"].set("mesh");
    for(int c = 0; c < nc; c++)
    {
      conduit::Node &n = n_field["values"].append();
      n.set(data[c]);
    }
  }
  return n_field;
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
clip_3d(conduit::Node &node, const std::string &name, bool do_inverse = true,
  const std::string &fieldname = std::string("test"), float clip_value = 0.5)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);
#if DEBUG_TEST
  handle_test(std::string("clip_") + name + "_orig", collection);
  {
    conduit::Node n_input;
    dray_collection_to_blueprint(collection, n_input);
    dray::BlueprintReader::save_blueprint(std::string("clip_") + name + "_orig", n_input);
  }
#endif
  // Filter.
  dray::ClipField clip;
  clip.set_clip_value(clip_value);
  clip.set_field(fieldname);
  clip.exclude_clip_field(fieldname == "test");

  dray::Collection output = clip.execute(collection);

#if DEBUG_TEST
  {
    conduit::Node n_output;
    dray_collection_to_blueprint(output, n_output);
    dray::BlueprintReader::save_blueprint(std::string("clip_") + name, n_output);
  }
#endif

  handle_test(std::string("clip_") + name, output);

  // Filter again, inverting the selection.
  if(do_inverse)
  {
    clip.set_invert_clip(true);
    dray::Collection output2 = clip.execute(collection);
    handle_test(std::string("clip_inv_") + name, output2);
  }
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
  handle_test(std::string("clip_2_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_2_plane_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_3_plane(conduit::Node &node, const std::string &name, bool multiplane)
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
  clip.SetMultiPlane(multiplane);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_3_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_3_plane_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
void
clip_box_plane(conduit::Node &node, const std::string &name)
{
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(node);
  collection.add_domain(domain);

  // Filter.
  dray::Clip clip;
  dray::AABB<3> bounds;
  bounds.m_ranges[0].set_range(-5., 5.);
  bounds.m_ranges[1].set_range(-2., 2.);
  bounds.m_ranges[2].set_range(-5., 5.);
  clip.SetBoxClip(bounds);
  dray::Collection output = clip.execute(collection);
  handle_test(std::string("clip_box_plane_") + name, output);

  // Filter again, inverting the selection.
  clip.SetInvertClip(true);
  dray::Collection output2 = clip.execute(collection);
  handle_test(std::string("clip_box_plane_inv_") + name, output2);
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_2_2_2_noclip)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             2,
                                             2,
                                             2,
                                             data);
  /*

   *----*
   |    |
   |    |
   *----*
   0    0
   */
  double values[] = {
    0., 0.,
    0., 0.,

    0., 0.,
    0., 0.
  };

  // Add another field.
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/topology"] = "mesh";
  data["fields/test/values"].set_external(values, 2*2*2);

  clip_3d(data, "hexs_2_2_2_noclip", false);
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_3_2_2_noclip)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  const int nx = 3, ny = 2, nz = 2;
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             nx,
                                             ny,
                                             nz,
                                             data);
  /*

   *----*----*
   |    |    |
   |    |    |
   *----*----*
   0    0    0
   */
  double values[] = {
    0., 0.,0.,
    0., 0.,0.,

    0., 0.,0.,
    0., 0.,0.
  };

  // Add another field.
  data["fields/test/topology"] = "mesh";
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/values"].set_external(values, 3*2*2);

  // Add another vector field.
  const int constant = 12;
  const int nc = 3;
  data["fields/vec"] = make_simple_field(constant, nx, ny, nz, nc);

  clip_3d(data, "hexs_3_2_2_noclip", false);
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_3_2_2_corner)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  const int nx = 3, ny = 2, nz = 2;
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             nx,
                                             ny,
                                             nz,
                                             data);
  /*
   0    0    1
   *----*----*
   |    |    |
   |    |    |
   *----*----*
   0    0    0
   */
  double values[] = {
    0., 0.,0.,
    0., 0.,1.,

    0., 0.,0.,
    0., 0.,1.
  };

  // Add another field.
  data["fields/test/topology"] = "mesh";
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/values"].set_external(values, 3*2*2);

  // Add another vector field.
  const int constant = 12;
  const int nc = 3;

  data["fields/vec"] = make_simple_field(constant, nx, ny, nz, nc);

  clip_3d(data, "hexs_3_2_2_corner");
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_3_3_2_hole)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             3,
                                             3,
                                             2,
                                             data);
  /*
   0    0    0
   *----*----*
   |    |    |
   |   1|    |     Make a hole in the middle
  0*----*----* 0
   |    |    |
   |    |    |
   *----*----*
   0    0    0
   */
  double values[] = {
    0., 0.,0.,
    0., 1.,0.,
    0., 0.,0.,

    0., 0.,0.,
    0., 1.,0.,
    0., 0.,0.
  };

  // Add another field.
  data["fields/test/topology"] = "mesh";
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/values"].set_external(values, 3*3*2);

  clip_3d(data, "hexs_3_3_2_hole");
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_3_2_2_vertical)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  const int nx = 3, ny = 2, nz = 2;
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             nx,
                                             ny,
                                             nz,
                                             data);
  /*

   *----*----*
   |    |    |
   |    |    |
   *----*----*
   1    0   -1

   */
  double values[] = {
    1., 0., -1.,
    1., 0., -1.,

    1., 0., -1.,
    1., 0., -1.,
  };

  // Add another field.
  data["fields/test/topology"] = "mesh";
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/values"].set_external(values, 3*2*2);

  // Add another vector field.
  const int constant = 12;
  const int nc = 3;
  data["fields/vec"] = make_simple_field(constant, nx, ny, nz, nc);

  clip_3d(data, "hexs_3_2_2_vertical");
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_3_3_3_vertical)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             3,
                                             3,
                                             3,
                                             data);
  /*

   *----*----*
   |    |    |
   |    |    |
   *----*----*
   |    |    |
   |    |    |
   *----*----*
   1    0   -1

   */
  double values[] = {
    1., 0., -1.,
    1., 0., -1.,
    1., 0., -1.,

    1., 0., -1.,
    1., 0., -1.,
    1., 0., -1.,

    1., 0., -1.,
    1., 0., -1.,
    1., 0., -1.     
  };

  // Add another field.
  data["fields/test/topology"] = "mesh";
  data["fields/test/association"] = "vertex";
  data["fields/test/type"] = "scalar";
  data["fields/test/values"].set_external(values, 3*3*3);

  clip_3d(data, "hexs_3_3_3_vertical");
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, hexs_braid)
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

  clip_3d(data, "hexs_braid", true, "braid", 4.8f);
}

//-----------------------------------------------------------------------------
TEST (dray_clip, hexs_sphere)
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

  clip_sphere(data, "hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip, hexs_1_plane)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM_SM,
                                             EXAMPLE_MESH_SIDE_DIM_SM,
                                             EXAMPLE_MESH_SIDE_DIM_SM,
                                             data);

  clip_1_plane(data, "hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip, hexs_2_plane)
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

  clip_2_plane(data, "hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clip, hexs_3_plane)
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

  clip_3_plane(data, "hexs", false);
}

//-----------------------------------------------------------------------------
TEST (dray_clip, hexs_box)
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

  clip_box_plane(data, "hexs");
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, tets_braid)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("tets",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  clip_3d(data, "tets_braid", true, "braid", 2.f);

  // TODO: Examine this because it makes scrambled geometry.
  //clip_3_plane(data, "tets_multiplane", true);
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, tets_1)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  // Make a simple tet mesh with 1 cell.
  double x[] = {0.f, 0.f, 1.f, 0.f};
  double y[] = {0.f, 0.f, 0.f, 1.f};
  double z[] = {0.f, 1.f, 0.f, 0.f};
  int conn[] = {0,1,2,3};

  conduit::Node data;
  data["coordsets/coords/type"] = "explicit";
  data["coordsets/coords/values/x"].set_external(x, sizeof(x)/sizeof(double));
  data["coordsets/coords/values/y"].set_external(y, sizeof(y)/sizeof(double));
  data["coordsets/coords/values/z"].set_external(z, sizeof(z)/sizeof(double));
  data["topologies/topology/coordset"] = "coords";
  data["topologies/topology/type"] = "unstructured";
  data["topologies/topology/elements/shape"] = "tet";
  data["topologies/topology/elements/connectivity"].set_external(conn, sizeof(conn)/sizeof(int));
  data["fields/height/topology"] = "topology";
  data["fields/height/association"] = "vertex";
  data["fields/height/values"].set_external(y, sizeof(y)/sizeof(double));

  //data.print();

  clip_3d(data, "tets_1", true, "height", 0.6);
}

//-----------------------------------------------------------------------------
TEST (dray_clipfield, tets_tiny)
{
#ifdef DEBUG_TEST
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
#endif
  // Make a simple tet mesh with 4 cells.
  double x[] = {0.f, 0.f, 1.f,  0.f, -1.f, 0.f};
  double y[] = {0.f, 0.f, 0.f,  0.f,  0.f, 1.f};
  double z[] = {0.f, 1.f, 0.f, -1.f,  0.f, 0.f};
  int conn[] = {0,1,2,5,  0,2,3,5,  0,3,4,5,  0,4,1,5};

  conduit::Node data;
  data["coordsets/coords/type"] = "explicit";
  data["coordsets/coords/values/x"].set_external(x, sizeof(x)/sizeof(double));
  data["coordsets/coords/values/y"].set_external(y, sizeof(y)/sizeof(double));
  data["coordsets/coords/values/z"].set_external(z, sizeof(z)/sizeof(double));
  data["topologies/topology/coordset"] = "coords";
  data["topologies/topology/type"] = "unstructured";
  data["topologies/topology/elements/shape"] = "tet";
  data["topologies/topology/elements/connectivity"].set_external(conn, sizeof(conn)/sizeof(int));
  data["fields/height/topology"] = "topology";
  data["fields/height/association"] = "vertex";
  data["fields/height/values"].set_external(y, sizeof(y)/sizeof(double));

  //data.print();

  clip_3d(data, "tets_tiny", true, "height", 0.6);
}
