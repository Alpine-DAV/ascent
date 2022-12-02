#include "gtest/gtest.h"

#include <dray/data_model/collection.hpp>
#include <dray/filters/isovolume.hpp>

const int EXAMPLE_MESH_SIDE_DIM = 15;
const int EXAMPLE_MESH_SIDE_DIM_SM = 7;

// There are a couple macros used by dray_test_utils
//#define DEBUG_TEST
//#define GENERATE_BASELINES
const std::string DRAY_TEST_NAME("isovolume");
#include "dray_test_utils.hpp"

// Writes the inputs and outputs of each test to the current working directory.
// Useful for debugging with visit
// #define WRITE_CONDUIT_DATASETS

//-----------------------------------------------------------------------------
static void
isovolume_3d(const conduit::Node &dset, const std::string name, 
  const std::string fieldname = std::string("test"),
  const dray::Vec2f range = {-0.5, 0.5})
{
  // Convert from blueprint to dray
  dray::Collection collection;
  dray::DataSet domain = dray::BlueprintReader::blueprint_to_dray(dset);
  collection.add_domain(domain);

#ifdef WRITE_CONDUIT_DATASETS
  {
    conduit::Node n_input;
    dray_collection_to_blueprint(collection, n_input);
    dray::BlueprintReader::save_blueprint(std::string("isovolume_") + name + "_orig", n_input);
  }
#endif

  // Filter
  dray::Isovolume isovolume;
  isovolume.exclude_clip_field(fieldname == "test");
  isovolume.set_field(fieldname);
  isovolume.set_range(range);

  dray::Collection output = isovolume.execute(collection);

#ifdef WRITE_CONDUIT_DATASETS
  {
    conduit::Node n_output;
    dray_collection_to_blueprint(output, n_output);
    dray::BlueprintReader::save_blueprint(std::string("isovolume_") + name, n_output);
  }
#endif

  handle_test(std::string("isovolume_") + name, output);
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_2_2_2_noclip)
{
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

  isovolume_3d(data, "hexs_2_2_2_noclip");
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_3_2_2_noclip)
{
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

  isovolume_3d(data, "hexs_3_2_2_noclip");
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_3_2_2_corners)
{
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
  -1    0    0
   */
  double values[] = {
    -1., 0.,0.,
    0., 0.,1.,

    -1., 0.,0.,
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

  // NOTE: This test generates odd geometry for the bottom left plane (the x=0 y=0 corner)
  //       maybe this similar to the odd geometry referenced in the tets_braid test of
  //       t_dray_clipfield. I think it might be a rogue tet lookup case, the top right (x=1 y=1 corner)
  //       appears normal to me.
  isovolume_3d(data, "hexs_3_2_2_corners");
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_3_2_2_vertical)
{
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

  isovolume_3d(data, "hexs_3_2_2_vertical");
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_3_3_3_vertical)
{
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

  isovolume_3d(data, "hexs_3_3_3_vertical");
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, hexs_braid)
{
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  isovolume_3d(data, "hexs_braid", "braid", {-2.f, 2.f});
}

//-----------------------------------------------------------------------------
TEST (t_dray_isovolume, tets_braid)
{
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("tets",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  isovolume_3d(data, "tets_braid", "braid", {-2.f, 2.f});
}
