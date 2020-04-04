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

#if defined(ASCENT_VTKM_ENABLED)
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_blueprint_filters.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#include <runtimes/ascent_data_object.hpp>
#include <runtimes/ascent_dray_data_adapter.hpp>

#include <dray/dray.hpp>
#include <dray/data_set.hpp>
#include <dray/filters/mesh_boundary.hpp>

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/rendering/partial_renderer.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <vtkh/vtkh.hpp>
#include <vtkh/compositing/Compositor.hpp>
#include <vtkh/compositing/PartialCompositor.hpp>
#include <vtkh/compositing/PayloadCompositor.hpp>

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

dray::Framebuffer partials_to_framebuffer(const std::vector<vtkh::VolumePartial<float>> &input,
                                          const int width,
                                          const int height)
{
  dray::Framebuffer fb(width, height);
  fb.clear();
  const int size = input.size();
  dray::Vec<float,4> *colors = fb.colors().get_host_ptr();
  float *depths = fb.depths().get_host_ptr();
#ifdef ASCENT_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    const int id = input[i].m_pixel_id;
    colors[id][0] = input[i].m_pixel[0];
    colors[id][1] = input[i].m_pixel[1];
    colors[id][2] = input[i].m_pixel[2];
    colors[id][3] = input[i].m_alpha;
    depths[id] = input[i].m_depth;
  }
  return fb;
}

void
parse_params(const conduit::Node &params,
             DRayCollection *dcol,
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
  dray::AABB<3> bounds = dcol->get_global_bounds();
  camera.reset_to_bounds(bounds);

  std::vector<float> clipping(2);
  clipping[0] = 0.01f;
  clipping[1] = 1000.f;
  if(params.has_path("camera"))
  {
    const conduit::Node &n_camera = params["camera"];
    clipping = detail::parse_camera(n_camera, camera);
  }

  dray::Range scalar_range = dcol->get_global_range(field_name);
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

  image_name = params["image_prefix"].as_string();
  image_name = expand_family_name(image_name, cycle);
}


void convert_partials(std::vector<dray::Array<dray::VolumePartial>> &input,
                      std::vector<std::vector<vtkh::VolumePartial<float>>> &output)
{
  size_t total_size = 0;
  const int in_size = input.size();
  std::vector<size_t> offsets;
  offsets.resize(in_size);
  output.resize(1);

  for(size_t i = 0; i< in_size; ++i)
  {
    offsets[i] = total_size;
    total_size += input[i].size();
  }

  output[0].resize(total_size);

  for(size_t a = 0; a < in_size; ++a)
  {
    const dray::VolumePartial *partial_ptr = input[a].get_host_ptr_const();
    const size_t offset = offsets[a];
    const size_t size = input[a].size();
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const size_t index = offset + i;
      const dray::VolumePartial p = partial_ptr[i];
      output[0][index].m_pixel_id = p.m_pixel_id;
      output[0][index].m_depth = p.m_depth;

      output[0][index].m_pixel[0] = p.m_color[0];
      output[0][index].m_pixel[1] = p.m_color[1];
      output[0][index].m_pixel[2] = p.m_color[2];

      output[0][index].m_alpha = p.m_color[3];
    }
  }

}

vtkh::PayloadImage * convert(dray::ScalarBuffer &result)
{
  const int num_fields = result.m_scalars.size();
  const int payload_size = num_fields * sizeof(float);
  vtkm::Bounds bounds;
  bounds.X.Min = 1;
  bounds.Y.Min = 1;
  bounds.X.Max = result.m_width;
  bounds.Y.Max = result.m_height;
  const size_t size = result.m_width * result.m_height;

  vtkh::PayloadImage *image = new vtkh::PayloadImage(bounds, payload_size);
  unsigned char *loads = &image->m_payloads[0];

  const float* dbuffer = result.m_depths.get_host_ptr();
  memcpy(&image->m_depths[0], dbuffer, sizeof(float) * size);
  // copy scalars into payload
  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    float* buffer = result.m_scalars[i].get_host_ptr();
    buffers.push_back(buffer);
  }
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(loads + offset, &buffers[i][x], sizeof(float));
    }
  }
  return image;
}

dray::ScalarBuffer  convert(vtkh::PayloadImage &image, std::vector<std::string> &names)
{
  dray::ScalarBuffer result;
  result.m_names = names;
  const int num_fields = names.size();

  const int dx  = image.m_bounds.X.Max - image.m_bounds.X.Min + 1;
  const int dy  = image.m_bounds.Y.Max - image.m_bounds.Y.Min + 1;
  const int size = dx * dy;

  result.m_width = dx;
  result.m_height = dy;

  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    dray::Array<float> array;
    array.resize(size);
    result.m_scalars.push_back(array);
    float* buffer = result.m_scalars[i].get_host_ptr();
    buffers.push_back(buffer);
  }

  const unsigned char *loads = &image.m_payloads[0];
  const size_t payload_size = image.m_payload_bytes;

  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(&buffers[i][x], loads + offset, sizeof(float));
    }
  }

  //
  result.m_depths.resize(size);
  float* dbuffer = result.m_depths.get_host_ptr();
  memcpy(dbuffer, &image.m_depths[0], sizeof(float) * size);

  return result;
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("field");
    valid_paths.push_back("image_prefix");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("log_scale");

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

    DRayCollection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    dcol->mpi_comm(comm_id);
    bool is_3d = dcol->topo_dims() == 3;

    DRayCollection faces = dcol->boundary();
    faces.mpi_comm(comm_id);

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

    const int num_domains = faces.m_domains.size();

    for(int i = 0; i < num_domains; ++i)
    {

      dray::Array<dray::Vec<dray::float32,4>> color_buffer;
      std::shared_ptr<dray::Surface> surface =
        std::make_shared<dray::Surface>(faces.m_domains[i]);

      surface->field(field_name);
      surface->color_map(color_map);
      surface->line_thickness(line_thickness);
      surface->line_color(vcolor);
      surface->draw_mesh(draw_mesh);

      dray::Renderer renderer;
      renderer.add(surface);
      renderer.use_lighting(is_3d);

      dray::Framebuffer fb = renderer.render(camera);

      std::vector<float> clipping(2);
      clipping[0] = 0.01f;
      clipping[1] = 1000.f;
      dray::Array<float32> depth = camera.gl_depth(fb.depths(), clipping[0], clipping[1]);
      depth_buffers.push_back(depth);
      color_buffers.push_back(fb.colors());
    }

#ifdef ASCENT_MPI_ENABLED
    vtkh::SetMPICommHandle(comm_id);
#endif

    vtkh::Compositor compositor;
    compositor.SetCompositeMode(vtkh::Compositor::Z_BUFFER_SURFACE);

    for(int i = 0; i < num_domains; ++i)
    {
      const float * cbuffer =
        reinterpret_cast<const float*>(color_buffers[i].get_host_ptr_const());
      compositor.AddImage(cbuffer,
                          depth_buffers[i].get_host_ptr_const(),
                          camera.get_width(),
                          camera.get_height());
    }
    vtkh::Image result = compositor.Composite();

    if(vtkh::GetMPIRank() == 0)
    {
      const float bg_color[4] = {1.f, 1.f, 1.f, 1.f};
      result.CompositeBackground(bg_color);
      PNGEncoder encoder;
      encoder.Encode(&result.m_pixels[0], camera.get_width(), camera.get_height());
      image_name = output_dir(image_name, graph());
      encoder.Save(image_name + ".png");
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("field");
    valid_paths.push_back("image_prefix");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
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

    DRayCollection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    dcol->mpi_comm(comm_id);

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

    dray::AABB<3> bounds = dcol->get_global_bounds();

    std::vector<dray::Array<dray::Vec<dray::float32,4>>> color_buffers;
    std::vector<dray::Array<dray::float32>> depth_buffers;

    using Vec3f = dray::Vec<float,3>;
    Vec3f x_normal({1.f, 0.f, 0.f});
    Vec3f y_normal({0.f, 1.f, 0.f});
    Vec3f z_normal({0.f, 0.f, 1.f});

    std::vector<Vec3f> points = detail::planes(params(), bounds);

    const int num_domains = dcol->m_domains.size();
    for(int i = 0; i < num_domains; ++i)
    {
      std::shared_ptr<dray::SlicePlane> slicer_x
        = std::make_shared<dray::SlicePlane>(dcol->m_domains[i]);

      std::shared_ptr<dray::SlicePlane> slicer_y
        = std::make_shared<dray::SlicePlane>(dcol->m_domains[i]);

      std::shared_ptr<dray::SlicePlane> slicer_z
        = std::make_shared<dray::SlicePlane>(dcol->m_domains[i]);

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
      renderer.add(slicer_x);
      renderer.add(slicer_y);
      renderer.add(slicer_z);

      dray::Framebuffer fb = renderer.render(camera);

      std::vector<float> clipping(2);
      clipping[0] = 0.01f;
      clipping[1] = 1000.f;
      dray::Array<float32> depth = camera.gl_depth(fb.depths(), clipping[0], clipping[1]);
      depth_buffers.push_back(depth);
      color_buffers.push_back(fb.colors());
    }

#ifdef ASCENT_MPI_ENABLED
    vtkh::SetMPICommHandle(comm_id);
#endif
    vtkh::Compositor compositor;
    compositor.SetCompositeMode(vtkh::Compositor::Z_BUFFER_SURFACE);

    for(int i = 0; i < num_domains; ++i)
    {
      const float * cbuffer =
        reinterpret_cast<const float*>(color_buffers[i].get_host_ptr_const());
      compositor.AddImage(cbuffer,
                          depth_buffers[i].get_host_ptr_const(),
                          camera.get_width(),
                          camera.get_height());
    }

    vtkh::Image result = compositor.Composite();

    if(vtkh::GetMPIRank() == 0)
    {
      const float bg_color[4] = {1.f, 1.f, 1.f, 1.f};
      result.CompositeBackground(bg_color);
      PNGEncoder encoder;
      encoder.Encode(&result.m_pixels[0], camera.get_width(), camera.get_height());
      image_name = output_dir(image_name, graph());
      encoder.Save(image_name + ".png");
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("field");
    valid_paths.push_back("image_prefix");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("log_scale");

    // filter knobs
    res &= check_numeric("samples",params, info, false);
    res &= check_string("use_lighing",params, info, false);

    valid_paths.push_back("samples");
    valid_paths.push_back("use_lighting");

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
DRayVolume::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("dray 3slice input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);

    DRayCollection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    dcol->mpi_comm(comm_id);

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

    dray::AABB<3> bounds = dcol->get_global_bounds();

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

    std::vector<dray::Array<dray::VolumePartial>> dom_partials;

    dray::PointLight plight;
    plight.m_pos = { 1.2f, -0.15f, 0.4f };
    plight.m_amb = { 1.0f, 1.0f, 1.f };
    plight.m_diff = { 0.5f, 0.5f, 0.5f };
    plight.m_spec = { 0.0f, 0.0f, 0.0f };
    plight.m_spec_pow = 90.0;
    dray::Array<dray::PointLight> lights;
    lights.resize(1);
    dray::PointLight *l_ptr = lights.get_host_ptr();
    l_ptr[0] = plight;

    const int num_domains = dcol->m_domains.size();
    for(int i = 0; i < num_domains; ++i)
    {
      std::shared_ptr<dray::PartialRenderer> volume
        = std::make_shared<dray::PartialRenderer>(dcol->m_domains[i]);

      dray::Array<dray::Ray> rays;
      camera.create_rays (rays);

      volume->samples(samples,bounds);
      volume->use_lighting(use_lighting);
      volume->field(field_name);
      volume->color_map() = color_map;
      dray::Array<dray::VolumePartial> partials = volume->integrate(rays, lights);
      dom_partials.push_back(partials);

    }

    std::vector<std::vector<vtkh::VolumePartial<float>>> c_partials;
    detail::convert_partials(dom_partials, c_partials);

    std::vector<vtkh::VolumePartial<float>> result;

    vtkh::PartialCompositor<vtkh::VolumePartial<float>> compositor;
#ifdef ASCENT_MPI_ENABLED
    compositor.set_comm_handle(comm_id);
#endif
    compositor.composite(c_partials, result);
    if(vtkh::GetMPIRank() == 0)
    {
      dray::Framebuffer fb = detail::partials_to_framebuffer(result,
                                                             camera.get_width(),
                                                             camera.get_height());
      fb.composite_background();
      image_name = output_dir(image_name, graph());
      fb.save(image_name);
    }
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

    ignore_paths.push_back("camera");

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

    DRayCollection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    dcol->mpi_comm(comm_id);

    DRayCollection faces = dcol->boundary();
    faces.mpi_comm(comm_id);

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

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    dray::AABB<3> bounds = dcol->get_global_bounds();
    camera.reset_to_bounds(bounds);

    //std::cout<<camera.print()<<"\n";

    std::vector<float> clipping(2);
    clipping[0] = 0.01f;
    clipping[1] = 1000.f;
    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      clipping = detail::parse_camera(n_camera, camera);
    }

    std::vector<dray::ScalarBuffer> buffers;

    // basic sanity checking
    int min_p = std::numeric_limits<int>::max();
    int max_p = std::numeric_limits<int>::min();

    std::vector<std::string> field_names;
    vtkh::PayloadCompositor compositor;
    const int num_domains = dcol->m_domains.size();

    for(int i = 0; i < num_domains; ++i)
    {
      std::shared_ptr<dray::Surface> surface =
        std::make_shared<dray::Surface>(faces.m_domains[i]);
      dray::ScalarRenderer renderer(surface);
      renderer.field_names(faces.m_domains[i].fields());
      if(i == 0)
      {
        field_names = faces.m_domains[i].fields();
      }

      dray::ScalarBuffer buffer = renderer.render(camera);
      vtkh::PayloadImage *pimage = detail::convert(buffer);

      min_p = std::min(min_p, pimage->m_payload_bytes);
      max_p = std::max(max_p, pimage->m_payload_bytes);

      compositor.AddImage(*pimage);
      delete pimage;
    }

    if(min_p != max_p)
    {
      ASCENT_ERROR("VERY BAD "<<min_p<<" "<<max_p<<" contact someone.");
    }

    vtkh::PayloadImage final_image = compositor.Composite();
    conduit::Node *output = new conduit::Node();
    if(vtkh::GetMPIRank() == 0)
    {
      dray::ScalarBuffer final_result = detail::convert(final_image, field_names);
      conduit::Node &dom = output->append();
      final_result.to_node(dom);
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





