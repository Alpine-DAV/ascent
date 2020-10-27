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
/// file: ascent_runtime_dray_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_dray_filters.hpp"

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
#include <ascent_runtime_utils.hpp>
#include <ascent_png_encoder.hpp>
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
#include <dray/data_set.hpp>
#include <dray/filters/mesh_boundary.hpp>

#include <dray/collection.hpp>
#include <dray/filters/reflect.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/volume_balance.hpp>
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
dray::PointLight default_light(dray::Camera &camera)
{
  dray::Vec<float32,3> look_at = camera.get_look_at();
  dray::Vec<float32,3> pos = camera.get_pos();
  dray::Vec<float32,3> up = camera.get_up();
  up.normalize();
  dray::Vec<float32,3> look = look_at - pos;
  dray::float32 mag = look.magnitude();
  dray::Vec<float32,3> right = cross (look, up);
  right.normalize();

  dray::Vec<float32, 3> miner_up = cross (right, look);
  miner_up.normalize();
  dray::Vec<float32, 3> light_pos = pos + .1f * mag * miner_up;
  dray::PointLight light;
  light.m_pos = light_pos;
  return light;
}

void frame_buffer_to_node(dray::Framebuffer &fb, conduit::Node &mesh)
{
  mesh.reset();
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = fb.width();
  mesh["coordsets/coords/dims/j"] = fb.height();

  const int size = fb.width() * fb.height();

  mesh["topologies/topo/coordset"] = "coords";
  mesh["topologies/topo/type"] = "uniform";

  const std::string path = "fields/colors/";
  mesh[path + "association"] = "vertex";
  mesh[path + "topology"] = "topo";
  const float *colors =  (float*)fb.colors().get_host_ptr_const();
  mesh[path + "values"].set(colors, size*4);

  mesh["fields/depth/association"] = "vertex";
  mesh["fields/depth/topology"] = "topo";
  const float32 *depths = fb.depths().get_host_ptr_const();
  mesh["fields/depth/values"].set(depths, size);

  conduit::Node verify_info;
  bool ok = conduit::blueprint::mesh::verify(mesh,verify_info);
  if(!ok)
  {
    verify_info.print();
  }
}

dray::Collection boundary(dray::Collection &collection)
{
  dray::MeshBoundary bounder;
  return bounder.execute(collection);
}

std::string
dray_color_table_surprises(const conduit::Node &color_table)
{
  std::string surprises;

  std::vector<std::string> valid_paths;
  valid_paths.push_back("name");
  valid_paths.push_back("reverse");

  std::vector<std::string> ignore_paths;
  ignore_paths.push_back("control_points");

  surprises += surprise_check(valid_paths, ignore_paths, color_table);
  if(color_table.has_path("control_points"))
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

std::string
dray_load_balance_surprises(const conduit::Node &load_balance)
{
  std::string surprises;

  std::vector<std::string> valid_paths;
  valid_paths.push_back("enabled");
  valid_paths.push_back("factor");
  valid_paths.push_back("threshold");
  valid_paths.push_back("use_prefix");

  surprises += surprise_check(valid_paths, load_balance);

  return surprises;
}

std::vector<dray::Vec<float,3>>
planes(const conduit::Node &params, const dray::AABB<3> bounds)
{

  using Vec3f = dray::Vec<float,3>;
  Vec3f center = bounds.center();
  Vec3f x_point = center;
  Vec3f y_point = center;
  Vec3f z_point = center;


  const float eps = 1e-5; // ensure that the slice is always inside the data set
  if(params.has_path("x_offset"))
  {
    float offset = params["x_offset"].to_float32();
    std::max(-1.f, std::min(1.f, offset));
    float t = (offset + 1.f) / 2.f;
    t = std::max(0.f + eps, std::min(1.f - eps, t));
    x_point[0] = bounds.m_ranges[0].min() + t * (bounds.m_ranges[0].length());
  }

  if(params.has_path("y_offset"))
  {
    float offset = params["y_offset"].to_float32();
    std::max(-1.f, std::min(1.f, offset));
    float t = (offset + 1.f) / 2.f;
    t = std::max(0.f + eps, std::min(1.f - eps, t));
    y_point[1] = bounds.m_ranges[1].min() + t * (bounds.m_ranges[1].length());
  }

  if(params.has_path("z_offset"))
  {
    float offset = params["z_offset"].to_float32();
    std::max(-1.f, std::min(1.f, offset));
    float t = (offset + 1.f) / 2.f;
    t = std::max(0.f + eps, std::min(1.f - eps, t));
    z_point[2] = bounds.m_ranges[2].min() + t * (bounds.m_ranges[2].length());
  }
  std::vector<Vec3f> points;
  points.push_back(x_point);
  points.push_back(y_point);
  points.push_back(z_point);
  return points;
}

std::vector<float>
parse_camera(const conduit::Node camera_node, dray::Camera &camera)
{
  typedef dray::Vec<float,3> Vec3f;
  //
  // Get the optional camera parameters
  //
  if(camera_node.has_child("look_at"))
  {
      conduit::Node n;
      camera_node["look_at"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f look_at{float(coords[0]), float(coords[1]), float(coords[2])};
      camera.set_look_at(look_at);
  }

  if(camera_node.has_child("position"))
  {
      conduit::Node n;
      camera_node["position"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f position{float(coords[0]), float(coords[1]), float(coords[2])};
      camera.set_pos(position);
  }

  if(camera_node.has_child("up"))
  {
      conduit::Node n;
      camera_node["up"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f up{float(coords[0]), float(coords[1]), float(coords[2])};
      up.normalize();
      camera.set_up(up);
  }

  if(camera_node.has_child("fov"))
  {
      camera.set_fov(camera_node["fov"].to_float64());
  }


  // this is an offset from the current azimuth
  if(camera_node.has_child("azimuth"))
  {
      double azimuth = camera_node["azimuth"].to_float64();
      camera.azimuth(azimuth);
  }
  if(camera_node.has_child("elevation"))
  {
      double elevation = camera_node["elevation"].to_float64();
      camera.elevate(elevation);
  }

  if(camera_node.has_child("zoom"))
  {
      float zoom = camera_node["zoom"].to_float32();
      camera.set_zoom(zoom);
  }
  //
  // With a new potential camera position. We need to reset the
  // clipping plane as not to cut out part of the data set
  //

  // clipping defaults
  std::vector<float> clipping(2);
  clipping[0] = 0.01f;
  clipping[1] = 1000.f;
  if(camera_node.has_child("near_plane"))
  {
      clipping[0] = camera_node["near_plane"].to_float64();
  }

  if(camera_node.has_child("far_plane"))
  {
      clipping[1] = camera_node["far_plane"].to_float64();
  }
  return clipping;
}

dray::ColorTable
parse_color_table(const conduit::Node &color_table_node)
{
  // default name
  std::string color_map_name = "cool2warm";

  if(color_table_node.number_of_children() == 0)
  {
    ASCENT_INFO("Color table node is empty (no children). Defaulting to "
                <<color_map_name);
  }

  bool name_provided = false;
  if(color_table_node.has_child("name"))
  {
    std::string name = color_table_node["name"].as_string();
    name_provided = true;
    std::vector<std::string> valid_names = dray::ColorTable::get_presets();
    auto loc = find(valid_names.begin(), valid_names.end(), name);
    if(loc != valid_names.end())
    {
      color_map_name = name;
    }
    else
    {
      std::stringstream ss;
      ss<<"[";
      for(int i = 0; i < valid_names.size(); ++i)
      {
        ss<<valid_names[i];
        if(i==valid_names.size()-1)
        {
          ss<<"]";
        }
        else
        {
          ss<<",";
        }
      }
      ASCENT_INFO("Invalid color table name '"<<name
                  <<"'. Defaulting to "<<color_map_name
                  <<". known names: "<<ss.str());
    }
  }

  dray::ColorTable color_table(color_map_name);

  if(color_table_node.has_child("control_points"))
  {
    bool clear = false;
    bool clear_alphas = false;
    // check to see if we have rgb points and clear the table
    NodeConstIterator itr = color_table_node.fetch("control_points").children();
    while(itr.has_next())
    {
        const Node &peg = itr.next();
        if (peg["type"].as_string() == "rgb")
        {
          clear = true;
        }
        if (peg["type"].as_string() == "alpha")
        {
          clear_alphas = true;
        }
    }

    if(clear && !name_provided)
    {
      color_table.clear_colors();
    }

    if(clear_alphas)
    {
      color_table.clear_alphas();
    }

    itr = color_table_node.fetch("control_points").children();
    while(itr.has_next())
    {
        const Node &peg = itr.next();
        if(!peg.has_child("position"))
        {
            ASCENT_WARN("Color map control point must have a position");
        }

        float64 position = peg["position"].to_float64();

        if(position > 1.0 || position < 0.0)
        {
              ASCENT_WARN("Cannot add color map control point position "
                            << position
                            << ". Must be a normalized scalar.");
        }

        if (peg["type"].as_string() == "rgb")
        {
            conduit::Node n;
            peg["color"].to_float32_array(n);
            const float *color = n.as_float32_ptr();

            dray::Vec<float,3> ecolor({color[0], color[1], color[2]});

            for(int i = 0; i < 3; ++i)
            {
              ecolor[i] = std::min(1.f, std::max(ecolor[i], 0.f));
            }

            color_table.add_point(position, ecolor);
        }
        else if (peg["type"].as_string() == "alpha")
        {
            float alpha = peg["alpha"].to_float32();
            alpha = std::min(1.f, std::max(alpha, 0.f));
            color_table.add_alpha(position, alpha);
        }
        else
        {
            ASCENT_WARN("Unknown color table control point type " << peg["type"].as_string()<<
                        "\nValid types are 'alpha' and 'rgb'");
        }
    }
  }

  if(color_table_node.has_child("reverse"))
  {
    if(color_table_node["reverse"].as_string() == "true")
    {
      color_table.reverse();
    }
  }
  return color_table;
}

void
parse_params(const conduit::Node &params,
             dray::Collection *dcol,
             const conduit::Node *meta,
             dray::Camera &camera,
             dray::ColorMap &color_map,
             std::string &field_name,
             std::string &image_name)
{
  field_name = params["field"].as_string();

  int width  = 512;
  int height = 512;

  if(params.has_path("image_width"))
  {
    width = params["image_width"].to_int32();
  }

  if(params.has_path("image_height"))
  {
    height = params["image_height"].to_int32();
  }

  camera.set_width(width);
  camera.set_height(height);
  dray::AABB<3> bounds = dcol->bounds();
  camera.reset_to_bounds(bounds);

  std::vector<float> clipping(2);
  clipping[0] = 0.01f;
  clipping[1] = 1000.f;
  if(params.has_path("camera"))
  {
    const conduit::Node &n_camera = params["camera"];
    clipping = detail::parse_camera(n_camera, camera);
  }

  dray::Range scalar_range = dcol->range(field_name);
  dray::Range range;
  if(params.has_path("min_value"))
  {
    range.include(params["min_value"].to_float32());
  }
  else
  {
    range.include(scalar_range.min());
  }

  if(params.has_path("max_value"))
  {
    range.include(params["max_value"].to_float32());
  }
  else
  {
    range.include(scalar_range.max());
  }

  color_map.scalar_range(range);

  bool log_scale = false;
  if(params.has_path("log_scale"))
  {
    if(params["log_scale"].as_string() == "true")
    {
      log_scale = true;
    }
  }

  color_map.log_scale(log_scale);

  if(params.has_path("color_table"))
  {
    color_map.color_table(parse_color_table(params["color_table"]));
  }

  int cycle = 0;

  if(meta->has_path("cycle"))
  {
    cycle = (*meta)["cycle"].as_int32();
  }

  if(params.has_path("image_prefix"))
  {
    image_name = params["image_prefix"].as_string();
    image_name = expand_family_name(image_name, cycle);
  }
}

}; // namespace detail

//-----------------------------------------------------------------------------
DRayPseudocolor::DRayPseudocolor()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayPseudocolor::~DRayPseudocolor()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayPseudocolor::declare_interface(Node &i)
{
    i["type_name"]   = "dray_pseudocolor";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
DRayPseudocolor::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_string("field",params, info, true);
    res &= check_string("image_prefix",params, info, true);
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
    valid_paths.push_back("draw_mesh");
    valid_paths.push_back("line_thickness");
    valid_paths.push_back("line_color");
    res &= check_numeric("line_color",params, info, false);
    res &= check_numeric("line_thickness",params, info, false);
    res &= check_string("draw_mesh",params, info, false);

    ignore_paths.push_back("camera");
    ignore_paths.push_back("color_table");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::dray_color_table_surprises(params["color_table"]);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRayPseudocolor::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray pseudocolor input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    bool is_3d = dcol->topo_dims() == 3;

    dray::Collection faces = detail::boundary(*dcol);

    dray::Camera camera;
    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::string image_name;
    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    detail::parse_params(params(),
                         &faces,
                         meta,
                         camera,
                         color_map,
                         field_name,
                         image_name);

    bool draw_mesh = false;
    if(params().has_path("draw_mesh"))
    {
      if(params()["draw_mesh"].as_string() == "true")
      {
        draw_mesh = true;
      }
    }
    float line_thickness = 0.05f;
    if(params().has_path("line_thickness"))
    {
      line_thickness = params()["line_thickness"].to_float32();
    }

    dray::Vec<float,4> vcolor = {0.f, 0.f, 0.f, 1.f};
    if(params().has_path("line_color"))
    {
      conduit::Node n;
      params()["line_color"].to_float32_array(n);
      if(n.dtype().number_of_elements() != 4)
      {
        ASCENT_ERROR("line_color is expected to be 4 floating "
                     "point values (RGBA)");
      }
      const float32 *color = n.as_float32_ptr();
      vcolor[0] = color[0];
      vcolor[1] = color[1];
      vcolor[2] = color[2];
      vcolor[3] = color[3];
    }

    std::vector<dray::Array<dray::Vec<dray::float32,4>>> color_buffers;
    std::vector<dray::Array<dray::float32>> depth_buffers;

    dray::Array<dray::Vec<dray::float32,4>> color_buffer;

    std::shared_ptr<dray::Surface> surface = std::make_shared<dray::Surface>(faces);
    surface->field(field_name);
    surface->color_map(color_map);
    surface->line_thickness(line_thickness);
    surface->line_color(vcolor);
    surface->draw_mesh(draw_mesh);

    dray::Renderer renderer;
    renderer.add(surface);
    renderer.use_lighting(is_3d);
    bool annotations = true;
    if(params().has_path("annotations"))
    {
      annotations = params()["annotations"].as_string() != "false";
    }
    renderer.screen_annotations(annotations);
    dray::Framebuffer fb = renderer.render(camera);

    if(dray::dray::mpi_rank() == 0)
    {
      fb.composite_background();
      image_name = output_dir(image_name, graph());
      fb.save(image_name);
    }

}

//-----------------------------------------------------------------------------
DRay3Slice::DRay3Slice()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRay3Slice::~DRay3Slice()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRay3Slice::declare_interface(Node &i)
{
    i["type_name"]   = "dray_3slice";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
DRay3Slice::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_string("field",params, info, true);
    res &= check_string("image_prefix",params, info, true);
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
    valid_paths.push_back("annotations");
    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("log_scale");

    // filter knobs
    res &= check_numeric("x_offset",params, info, false);
    res &= check_numeric("y_offset",params, info, false);
    res &= check_numeric("z_offset",params, info, false);

    valid_paths.push_back("x_offset");
    valid_paths.push_back("y_offset");
    valid_paths.push_back("z_offset");

    ignore_paths.push_back("camera");
    ignore_paths.push_back("color_table");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::dray_color_table_surprises(params["color_table"]);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRay3Slice::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray 3slice input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Camera camera;
    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::string image_name;
    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    detail::parse_params(params(),
                         dcol,
                         meta,
                         camera,
                         color_map,
                         field_name,
                         image_name);

    dray::AABB<3> bounds = dcol->bounds();

    using Vec3f = dray::Vec<float,3>;
    Vec3f x_normal({1.f, 0.f, 0.f});
    Vec3f y_normal({0.f, 1.f, 0.f});
    Vec3f z_normal({0.f, 0.f, 1.f});

    std::vector<Vec3f> points = detail::planes(params(), bounds);

    std::shared_ptr<dray::SlicePlane> slicer_x
      = std::make_shared<dray::SlicePlane>(*dcol);

    std::shared_ptr<dray::SlicePlane> slicer_y
      = std::make_shared<dray::SlicePlane>(*dcol);

    std::shared_ptr<dray::SlicePlane> slicer_z
      = std::make_shared<dray::SlicePlane>(*dcol);

    slicer_x->field(field_name);
    slicer_y->field(field_name);
    slicer_z->field(field_name);

    slicer_x->color_map(color_map);
    slicer_y->color_map(color_map);
    slicer_z->color_map(color_map);

    slicer_x->point(points[0]);
    slicer_x->normal(x_normal);

    slicer_y->point(points[1]);
    slicer_y->normal(y_normal);

    slicer_z->point(points[2]);
    slicer_z->normal(z_normal);

    dray::Renderer renderer;

    bool annotations = true;
    if(params().has_path("annotations"))
    {
      annotations = params()["annotations"].as_string() != "false";
    }
    renderer.screen_annotations(annotations);

    renderer.add(slicer_x);
    renderer.add(slicer_y);
    renderer.add(slicer_z);

    dray::Framebuffer fb = renderer.render(camera);

    if(dray::dray::mpi_rank() == 0)
    {
      fb.composite_background();
      image_name = output_dir(image_name, graph());
      fb.save(image_name);
    }
}

//-----------------------------------------------------------------------------
DRayVolume::DRayVolume()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayVolume::~DRayVolume()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayVolume::declare_interface(Node &i)
{
    i["type_name"]   = "dray_volume";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
DRayVolume::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_string("field",params, info, true);
    res &= check_string("image_prefix",params, info, true);
    res &= check_numeric("min_value",params, info, false);
    res &= check_numeric("max_value",params, info, false);
    res &= check_numeric("image_width",params, info, false);
    res &= check_numeric("image_height",params, info, false);
    res &= check_string("log_scale",params, info, false);
    res &= check_string("annotations",params, info, false);
    res &= check_string("load_balancing",params, info, false);

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
    ignore_paths.push_back("load_balancing");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::dray_color_table_surprises(params["color_table"]);
    }

    if(params.has_path("load_balancing"))
    {
      surprises += detail::dray_load_balance_surprises(params["load_balancing"]);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRayVolume::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray 3slice input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection dataset = *dcol;

    dray::Camera camera;

    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::string image_name;
    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    detail::parse_params(params(),
                         dcol,
                         meta,
                         camera,
                         color_map,
                         field_name,
                         image_name);

    if(color_map.color_table().number_of_alpha_points() == 0)
    {
      color_map.color_table().add_alpha (0.f, 0.00f);
      color_map.color_table().add_alpha (0.1f, 0.00f);
      color_map.color_table().add_alpha (0.3f, 0.05f);
      color_map.color_table().add_alpha (0.4f, 0.21f);
      color_map.color_table().add_alpha (1.0f, 0.9f);
    }

    bool use_lighting = false;
    if(params().has_path("use_lighting"))
    {
      if(params()["use_lighting"].as_string() == "true")
      {
        use_lighting = true;
      }
    }

    int samples = 100;
    if(params().has_path("samples"))
    {
      samples = params()["samples"].to_int32();
    }

    if(params().has_path("load_balancing"))
    {
      const conduit::Node &load = params()["load_balancing"];
      bool enabled = true;
      if(load.has_path("enabled") &&
         load["enabled"].as_string() == "false")
      {
        enabled = false;
      }

      if(enabled)
      {
        float piece_factor = 0.75f;
        float threshold = 2.0f;
        bool prefix = true;
        if(load.has_path("factor"))
        {
          piece_factor = load["factor"].to_float32();
        }
        if(load.has_path("threshold"))
        {
          threshold = load["threshold"].to_float32();
        }

        if(load.has_path("use_prefix"))
        {
          prefix = load["use_prefix"].as_string() != "false";
        }
        dray::VolumeBalance balancer;
        balancer.threshold(threshold);
        balancer.prefix_balancing(prefix);
        balancer.piece_factor(piece_factor);

        dataset = balancer.execute(dataset, camera, samples);

      }

    }

    std::shared_ptr<dray::Volume> volume
      = std::make_shared<dray::Volume>(dataset);

    volume->color_map() = color_map;
    volume->samples(samples);
    volume->field(field_name);
    dray::Renderer renderer;
    renderer.volume(volume);

    bool annotations = true;
    if(params().has_path("annotations"))
    {
      annotations = params()["annotations"].as_string() != "false";
    }
    renderer.screen_annotations(annotations);

    dray::Framebuffer fb = renderer.render(camera);

    if(dray::dray::mpi_rank() == 0)
    {
      fb.composite_background();
      image_name = output_dir(image_name, graph());
      fb.save(image_name);
    }

}

//-----------------------------------------------------------------------------
DRayReflect::DRayReflect()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayReflect::~DRayReflect()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayReflect::declare_interface(Node &i)
{
    i["type_name"]   = "dray_reflect";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DRayReflect::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_numeric("point/x",params, info, true);
    res &= check_numeric("point/y",params, info, true);
    res &= check_numeric("point/z",params, info, false);
    res &= check_numeric("normal/x",params, info, true);
    res &= check_numeric("normal/y",params, info, true);
    res &= check_numeric("normal/z",params, info, false);

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("point/x");
    valid_paths.push_back("point/y");
    valid_paths.push_back("point/z");
    valid_paths.push_back("normal/x");
    valid_paths.push_back("normal/y");
    valid_paths.push_back("normal/z");


    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRayReflect::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray reflect input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Vec<float,3> point = {0.f, 0.f, 0.f};
    point[0] = params()["point/x"].to_float32();
    point[1] = params()["point/y"].to_float32();
    if(params().has_path("point/z"))
    {
      point[2] = params()["point/z"].to_float32();
    }

    dray::Vec<float,3> normal= {0.f, 1.f, 0.f};
    normal[0] = params()["normal/x"].to_float32();
    normal[1] = params()["normal/y"].to_float32();
    if(params().has_path("normal/z"))
    {
      normal[2] = params()["normal/z"].to_float32();
    }

    dray::Reflect reflector;
    reflector.plane(point,normal);

    dray::Collection output = reflector.execute(*dcol);

    for(int i = 0; i < dcol->size(); ++i)
    {
      dray::DataSet dset = dcol->domain(i);
      output.add_domain(dset);
    }

    dray::Collection *output_ptr = new dray::Collection();
    *output_ptr = output;

    DataObject *res =  new DataObject(output_ptr);
    set_output<DataObject>(res);
}


//-----------------------------------------------------------------------------
DRayProject2d::DRayProject2d()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayProject2d::~DRayProject2d()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayProject2d::declare_interface(Node &i)
{
    i["type_name"]   = "dray_project_2d";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DRayProject2d::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_numeric("image_width",params, info, false);
    res &= check_numeric("image_height",params, info, false);

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("fields");

    ignore_paths.push_back("camera");
    ignore_paths.push_back("fields");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRayProject2d::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray_project2d input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection faces = detail::boundary(*dcol);

    std::string image_name;

    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");
    int width  = 512;
    int height = 512;

    if(params().has_path("image_width"))
    {
      width = params()["image_width"].to_int32();
    }

    if(params().has_path("image_height"))
    {
      height = params()["image_height"].to_int32();
    }

    std::vector<std::string> field_selection;
    if(params().has_path("fields"))
    {
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("dray_project_2d  field selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("relay_io_save field selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
    }

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    dray::AABB<3> bounds = dcol->bounds();
    camera.reset_to_bounds(bounds);

    std::vector<float> clipping(2);
    clipping[0] = 0.01f;
    clipping[1] = 1000.f;
    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      clipping = detail::parse_camera(n_camera, camera);
    }

    std::vector<dray::ScalarBuffer> buffers;


    std::vector<std::string> field_names;

    std::shared_ptr<dray::Surface> surface = std::make_shared<dray::Surface>(faces);
    dray::ScalarRenderer renderer(surface);

    if(field_selection.size() == 0)
    {
      field_names = faces.domain(0).fields();
    }
    else
    {
      field_names = field_selection;
    }

    renderer.field_names(field_names);
    dray::ScalarBuffer sb = renderer.render(camera);

    conduit::Node *output = new conduit::Node();
    if(dray::dray::mpi_rank() == 0)
    {
      conduit::Node &dom = output->append();

      sb.to_node(dom);

      dom["state/domain_id"] = 0;

      int cycle = 0;

      if(meta->has_path("cycle"))
      {
        cycle = (*meta)["cycle"].as_int32();
      }
      dom["state/cycle"] = cycle;
    }

    DataObject *res =  new DataObject(output);
    set_output<DataObject>(res);

}

//-----------------------------------------------------------------------------
DRayProjectColors2d::DRayProjectColors2d()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayProjectColors2d::~DRayProjectColors2d()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayProjectColors2d::declare_interface(Node &i)
{
    i["type_name"]   = "dray_project_colors_2d";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DRayProjectColors2d::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();

    bool res = true;

    res &= check_numeric("image_width",params, info, false);
    res &= check_numeric("image_height",params, info, false);
    res &= check_string("field",params, info, true);

    res &= check_numeric("min_value",params, info, false);
    res &= check_numeric("max_value",params, info, false);
    res &= check_numeric("image_width",params, info, false);
    res &= check_numeric("image_height",params, info, false);
    res &= check_string("log_scale",params, info, false);

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    valid_paths.push_back("log_scale");
    valid_paths.push_back("field");

    ignore_paths.push_back("camera");
    ignore_paths.push_back("color_table");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::dray_color_table_surprises(params["color_table"]);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
DRayProjectColors2d::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray_project2d input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection faces = detail::boundary(*dcol);

    dray::Camera camera;
    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::string image_name;
    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    detail::parse_params(params(),
                         &faces,
                         meta,
                         camera,
                         color_map,
                         field_name,
                         image_name);


    const int num_domains = faces.size();

    dray::Framebuffer framebuffer (camera.get_width(), camera.get_height());
    std::shared_ptr<dray::Surface> surface = std::make_shared<dray::Surface>(faces);

    dray::Array<dray::PointLight> lights;
    lights.resize(1);
    dray::PointLight light = detail::default_light(camera);
    dray::PointLight* light_ptr = lights.get_host_ptr();
    light_ptr[0] = light;

    dray::Array<dray::Ray> rays;
    camera.create_rays (rays);

    conduit::Node* image_data = new conduit::Node();

    for(int i = 0; i < num_domains; ++i)
    {
      framebuffer.clear();
      surface->active_domain(i);
      surface->field(field_name);
      dray::Array<dray::RayHit> hits = surface->nearest_hit(rays);
      dray::Array<dray::Fragment> fragments = surface->fragments(hits);
      surface->shade(rays, hits, fragments, lights, framebuffer);
      conduit::Node &img = image_data->append();
      detail::frame_buffer_to_node(framebuffer, img);
    }

    DataObject *res =  new DataObject(image_data);
    set_output<DataObject>(res);

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





