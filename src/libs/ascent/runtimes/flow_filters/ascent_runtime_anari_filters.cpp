//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_anari_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_anari_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_metadata.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_resources.hpp>
#include <png_utils/ascent_png_encoder.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#include <runtimes/ascent_data_object.hpp>

#include <dray/dray.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/filters/reflect.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/volume_balance.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/dray_exports.h>
#include <dray/transform_3d.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/io/blueprint_reader.hpp>

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

namespace detail
{

static std::string
dray_color_table_surprises(const conduit::Node &color_table)
{
  std::string surprises;

  std::vector<std::string> valid_paths;
  valid_paths.push_back("name");
  valid_paths.push_back("reverse");

  std::vector<std::string> ignore_paths;
  ignore_paths.push_back("control_points");

  surprises += surprise_check(valid_paths, ignore_paths, color_table);
  if (color_table.has_path("control_points"))
  {
    std::vector<std::string> c_valid_paths;
    c_valid_paths.push_back("type");
    c_valid_paths.push_back("alpha");
    c_valid_paths.push_back("color");
    c_valid_paths.push_back("position");

    const conduit::Node &control_points = color_table["control_points"];
    const int num_points = control_points.number_of_children();
    for(int i = 0; i < num_points; ++i)
    {
      const conduit::Node &point = control_points.child(i);
      surprises += surprise_check(c_valid_paths, point);
    }
  }

  return surprises;
}

void
parse_camera(const conduit::Node camera_node, dray::Camera &camera);

dray::ColorTable
parse_color_table(const conduit::Node &color_table_node);

static bool 
check_image_names(const conduit::Node &params, conduit::Node &info)
{
  bool res = true;
  // if (!params.has_path("image_prefix") && !params.has_path("camera/db_name"))
  // {
  //   res = false;
  //   info.append() = "Anari ray rendering paths must include either "
  //                   "a 'image_prefix' (if its a single image) or a "
  //                   "'camera/db_name' (if using a cinema camere)";
  // }
  // if (params.has_path("image_prefix") && params.has_path("camera/db_name"))
  // {
  //   res = false;
  //   info.append() = "Anari ray rendering paths cannot use both "
  //                   "a 'image_prefix' (if its a single image) and a "
  //                   "'camera/db_name' (if using a cinema camere)";
  // }
  return res;
}

static void
parse_params(const conduit::Node &params,
             dray::Collection *dcol,
             const conduit::Node *meta,
             std::vector<dray::Camera> &cameras,
             dray::ColorMap &color_map,
             std::string &field_name,
             std::vector<std::string> &image_names)
{
  field_name = params["field"].as_string();

  int width  = 512;
  int height = 512;

  if (params.has_path("image_width"))
  {
    width = params["image_width"].to_int32();
  }

  if (params.has_path("image_height"))
  {
    height = params["image_height"].to_int32();
  }

  dray::AABB<3> bounds = dcol->bounds();

  if (params.has_path("camera"))
  {
    bool is_cinema = false;

    const conduit::Node &n_camera = params["camera"];
    if (n_camera.has_path("type"))
    {
      if (n_camera["type"].as_string() == "cinema")
      {
        is_cinema = true;
      }
    }

    if (is_cinema)
    {
      ASCENT_ERROR("Cinema database not yet implemented for Anari");
    }
    else
    {
      dray::Camera camera;
      camera.set_width(width);
      camera.set_height(height);
      camera.reset_to_bounds(bounds);
      detail::parse_camera(n_camera, camera);
      cameras.push_back(camera);
    }
  }
  else // if we don't have camera params, we need to add a default camera
  {
    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    camera.reset_to_bounds(bounds);
    cameras.push_back(camera);
  }

  int cycle = 0;

  if (meta->has_path("cycle"))
  {
    cycle = (*meta)["cycle"].as_int32();
  }

  if (params.has_path("image_prefix"))
  {

    std::string image_name = params["image_prefix"].as_string();
    image_name = expand_family_name(image_name, cycle);
    image_name = output_dir(image_name);
    image_names.push_back(image_name);
  }

  dray::Range scalar_range = dcol->range(field_name);
  dray::Range range;
  if (params.has_path("min_value"))
  {
    range.include(params["min_value"].to_float32());
  }
  else
  {
    range.include(scalar_range.min());
  }

  if (params.has_path("max_value"))
  {
    range.include(params["max_value"].to_float32());
  }
  else
  {
    range.include(scalar_range.max());
  }

  color_map.scalar_range(range);

  bool log_scale = false;
  if (params.has_path("log_scale"))
  {
    if (params["log_scale"].as_string() == "true")
    {
      log_scale = true;
    }
  }

  color_map.log_scale(log_scale);

  if (params.has_path("color_table"))
  {
    color_map.color_table(parse_color_table(params["color_table"]));
  }
}

}; // namespace detail


//-----------------------------------------------------------------------------
AnariVolume::AnariVolume()
  :Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
AnariVolume::~AnariVolume()
{
  // empty
}

//-----------------------------------------------------------------------------
void
AnariVolume::declare_interface(Node &i)
{
    i["type_name"]   = "dray_volume";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
AnariVolume::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();

  bool res = true;

  res &= check_string("field",params, info, true);
  res &= detail::check_image_names(params, info);
  res &= check_numeric("min_value",params, info, false);
  res &= check_numeric("max_value",params, info, false);
  res &= check_numeric("image_width",params, info, false);
  res &= check_numeric("image_height",params, info, false);
  res &= check_string("log_scale",params, info, false);
  res &= check_string("annotations",params, info, false);

  std::vector<std::string> valid_paths;
  std::vector<std::string> ignore_paths;

  valid_paths.push_back("field");
  valid_paths.push_back("image_prefix");
  valid_paths.push_back("min_value");
  valid_paths.push_back("max_value");
  valid_paths.push_back("image_width");
  valid_paths.push_back("image_height");
  valid_paths.push_back("log_scale");
  valid_paths.push_back("annotations");

  // filter knobs
  res &= check_numeric("samples",params, info, false);
  res &= check_string("use_lighing",params, info, false);

  valid_paths.push_back("samples");
  valid_paths.push_back("use_lighting");

  ignore_paths.push_back("camera");
  ignore_paths.push_back("color_table");

  std::string surprises = surprise_check(valid_paths, ignore_paths, params);

  if (params.has_path("color_table"))
  {
    surprises += detail::dray_color_table_surprises(params["color_table"]);
  }

  if (surprises != "")
  {
    res = false;
    info["errors"].append() = surprises;
  }
  return res;
}

//-----------------------------------------------------------------------------
void
AnariVolume::execute()
{
  if (!input(0).check_type<DataObject>())
  {
    ASCENT_ERROR("dray 3slice input must be a DataObject");
  }

  DataObject *d_input = input<DataObject>(0);
  if (!d_input->is_valid())
  {
    return;
  }

  VTKHCollection *collection = d_input->as_vtkh_collection().get();

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection dataset = *dcol;

    std::vector<dray::Camera> cameras;

    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::vector<std::string> image_names;
    conduit::Node meta = Metadata::n_metadata;

    detail::parse_params(params(),
                         dcol,
                         &meta,
                         cameras,
                         color_map,
                         field_name,
                         image_names);

    if (color_map.color_table().number_of_alpha_points() == 0)
    {
      color_map.color_table().add_alpha (0.f, 0.00f);
      color_map.color_table().add_alpha (0.1f, 0.00f);
      color_map.color_table().add_alpha (0.3f, 0.05f);
      color_map.color_table().add_alpha (0.4f, 0.21f);
      color_map.color_table().add_alpha (1.0f, 0.9f);
    }

    bool use_lighting = false;
    if (params().has_path("use_lighting"))
    {
      if (params()["use_lighting"].as_string() == "true")
      {
        use_lighting = true;
      }
    }

    int samples = 100;
    if (params().has_path("samples"))
    {
      samples = params()["samples"].to_int32();
    }

    std::shared_ptr<dray::Volume> volume
      = std::make_shared<dray::Volume>(dataset);

    volume->color_map() = color_map;
    volume->samples(samples);
    volume->field(field_name);
    dray::Renderer renderer;
    renderer.volume(volume);

    bool annotations = true;
    if (params().has_path("annotations"))
    {
      annotations = params()["annotations"].as_string() != "false";
    }
    renderer.world_annotations(annotations);

    const int num_images = cameras.size();
    for(int i = 0; i < num_images; ++i)
    {
      dray::Camera &camera = cameras[i];
      dray::Framebuffer fb = renderer.render(camera);

      if (dray::dray::mpi_rank() == 0)
      {
        fb.composite_background();
        fb.save(image_names[i]);
      }
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
