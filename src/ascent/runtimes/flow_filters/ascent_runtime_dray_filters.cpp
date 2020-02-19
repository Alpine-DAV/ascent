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
#include <dray/rendering/partial_renderer.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <vtkh/vtkh.hpp>
#include <vtkh/rendering/Compositor.hpp>
#include <vtkh/rendering/PartialCompositor.hpp>

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
      vtkm::Float64 azimuth = camera_node["azimuth"].to_float64();
      camera.azimuth(azimuth);
  }
  if(camera_node.has_child("elevation"))
  {
      vtkm::Float64 elevation = camera_node["elevation"].to_float64();
      camera.elevate(elevation);
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

dray::ColorMap
parse_color_table(const conduit::Node &color_map_node)
{
  // default name
  std::string color_map_name = "cool to warm";

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
    if(is_valid_name(name))
    {
      color_map_name = name;
    }
    else
    {
      ASCENT_INFO("Invalid color table name '"<<name
                  <<"'. Defaulting to "<<color_map_name);
    }
  }

  vtkm::cont::ColorTable color_table(color_map_name);

  if(color_table_node.has_child("control_points"))
  {
    bool clear = false;
    // check to see if we have rgb points and clear the table
    NodeConstIterator itr = color_table_node.fetch("control_points").children();
    while(itr.has_next())
    {
        const Node &peg = itr.next();
        if (peg["type"].as_string() == "rgb")
        {
          clear = true;
          break;
        }
    }

    if(clear && !name_provided)
    {
      color_table.ClearColors();
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
            peg["color"].to_float64_array(n);
            const float64 *color = n.as_float64_ptr();

            vtkm::Vec<vtkm::Float64,3> ecolor(color[0], color[1], color[2]);

            for(int i = 0; i < 3; ++i)
            {
              ecolor[i] = std::min(1., std::max(ecolor[i], 0.));
            }

            color_table.AddPoint(position, ecolor);
        }
        else if (peg["type"].as_string() == "alpha")
        {
            float64 alpha = peg["alpha"].to_float64();
            alpha = std::min(1., std::max(alpha, 0.));
            color_table.AddPointAlpha(position, alpha);
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
      color_table.ReverseColors();
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

    if(! params.has_child("field") ||
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
        res = false;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
DRayPseudocolor::execute()
{
    std::cout<<"BN\n";
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

    std::string field_name = params()["field"].as_string();
    dray::Range scalar_range = dcol->get_global_range(field_name);

    bool draw_mesh = false;
    if(params().has_path("draw_mesh"))
    {
      if(params()["draw_mesh"].as_string() == "on")
      {
        draw_mesh = true;
      }
    }
    float line_thickness = 0.05f;
    if(params().has_path("line_thickness"))
    {
      line_thickness = params()["line_thickness"].to_float32();
    }


    DRayCollection faces = dcol->boundary();
    faces.mpi_comm(comm_id);

    dray::AABB<3> bounds = faces.get_global_bounds();
    int width  = 512;
    int height = 512;

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    camera.reset_to_bounds(bounds);

    std::vector<float> clipping(2);
    clipping[0] = 0.01f;
    clipping[1] = 1000.f;
    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      clipping = detail::parse_camera(n_camera, camera);
    }

    dray::ColorMap color_map("cool2warm");
    color_map.scalar_range(scalar_range);

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
      surface->draw_mesh(draw_mesh);

      dray::Renderer renderer;
      renderer.add(surface);

      dray::Framebuffer fb = renderer.render(camera);

      dray::Array<float32> depth = camera.gl_depth(fb.depths(), clipping[0], clipping[1]);
      depth_buffers.push_back(depth);
      color_buffers.push_back(fb.colors());
    }

    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    int cycle = 0;

    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    std::string image_name = "dray_surface_%06d";
    if(params().has_path("image_prefix"))
    {
      image_name = params()["image_prefix"].as_string();
    }

    image_name = expand_family_name(image_name, cycle);

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
                          width,
                          height);
    }
    vtkh::Image result = compositor.Composite();

    if(vtkh::GetMPIRank() == 0)
    {
      PNGEncoder encoder;
      encoder.Encode(&result.m_pixels[0], width, height);
      encoder.Save(image_name + ".png");
    }
    std::cout<<"xxxBN\n";
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("field");
    ignore_paths.push_back("camera");

    res &= check_numeric("x_offset",params, info, false);
    res &= check_numeric("y_offset",params, info, false);
    res &= check_numeric("z_offset",params, info, false);

    valid_paths.push_back("x_offset");
    valid_paths.push_back("y_offset");
    valid_paths.push_back("z_offset");


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
DRay3Slice::execute()
{
    std::cout<<"BN\n";
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

    dray::AABB<3> bounds = dcol->get_global_bounds();

    int width  = 512;
    int height = 512;

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    camera.reset_to_bounds(bounds);

    std::vector<float> clipping(2);
    clipping[0] = 0.01f;
    clipping[1] = 1000.f;
    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      clipping = detail::parse_camera(n_camera, camera);
    }
    std::string field_name = params()["field"].as_string();

    dray::ColorMap color_map("cool2warm");
    dray::Range scalar_range = dcol->get_global_range(field_name);
    color_map.scalar_range(scalar_range);

    std::vector<dray::Array<dray::Vec<dray::float32,4>>> color_buffers;
    std::vector<dray::Array<dray::float32>> depth_buffers;

    dray::PointLight plight;
    plight.m_pos = { 1.2f, -0.15f, 0.4f };
    plight.m_amb = { 1.0f, 1.0f, 1.f };
    plight.m_diff = { 0.5f, 0.5f, 0.5f };
    plight.m_spec = { 0.0f, 0.0f, 0.0f };
    plight.m_spec_pow = 90.0;

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
      renderer.add_light(plight);

      dray::Framebuffer fb = renderer.render(camera);

      dray::Array<float32> depth = camera.gl_depth(fb.depths(), clipping[0], clipping[1]);
      depth_buffers.push_back(depth);
      color_buffers.push_back(fb.colors());
    }

    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    int cycle = 0;

    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    std::string image_name = "dray_3slice_%06d";
    if(params().has_path("image_prefix"))
    {
      image_name = params()["image_prefix"].as_string();
    }

    image_name = expand_family_name(image_name, cycle);

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
                          width,
                          height);
    }
    vtkh::Image result = compositor.Composite();

    if(vtkh::GetMPIRank() == 0)
    {
      PNGEncoder encoder;
      encoder.Encode(&result.m_pixels[0], width, height);
      encoder.Save(image_name + ".png");
    }
    std::cout<<"xxxBN\n";
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

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;

    valid_paths.push_back("field");
    ignore_paths.push_back("camera");

    res &= check_numeric("samples",params, info, false);
    res &= check_numeric("width",params, info, false);
    res &= check_numeric("height",params, info, false);
    res &= check_numeric("min_value",params, info, false);
    res &= check_numeric("max_value",params, info, false);

    valid_paths.push_back("samples");
    valid_paths.push_back("width");
    valid_paths.push_back("height");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");

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
DRayVolume::execute()
{
    std::cout<<"BN\n";
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

    dray::AABB<3> bounds = dcol->get_global_bounds();

    int width  = 512;
    int height = 512;

    if(params().has_path("width"))
    {
      width = params()["width"].to_int32();
    }

    if(params().has_path("height"))
    {
      height = params()["height"].to_int32();
    }

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    camera.reset_to_bounds(bounds);

    std::vector<float> clipping(2);
    clipping[0] = 0.01f;
    clipping[1] = 1000.f;
    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      clipping = detail::parse_camera(n_camera, camera);
    }

    std::string field_name = params()["field"].as_string();

    dray::ColorMap color_map("cool2warm");

    color_map.color_table().add_alpha (0.f, 0.00f);
    color_map.color_table().add_alpha (0.1f, 0.00f);
    color_map.color_table().add_alpha (0.3f, 0.05f);
    color_map.color_table().add_alpha (0.4f, 0.21f);
    color_map.color_table().add_alpha (1.0f, 0.9f);

    dray::Range scalar_range = dcol->get_global_range(field_name);
    color_map.scalar_range(scalar_range);

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

      volume->field(field_name);
      volume->color_map() = color_map;
      dray::Array<dray::VolumePartial> partials = volume->integrate(rays, lights);
      dom_partials.push_back(partials);


    }

    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    int cycle = 0;

    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    std::string image_name = "dray_volume_%06d";
    if(params().has_path("image_prefix"))
    {
      image_name = params()["image_prefix"].as_string();
    }

    image_name = expand_family_name(image_name, cycle);

    std::vector<std::vector<vtkh::VolumePartial<float>>> c_partials;
    detail::convert_partials(dom_partials, c_partials);

    std::vector<vtkh::VolumePartial<float>> result;

    vtkh::PartialCompositor<vtkh::VolumePartial<float>> compositor;
#ifdef ASCENT_MPI_ENABLED
    compositor.set_comm_handle(comm_id);
#endif
    //compositor.set_background(m_background);
    std::cout<<"COMP\n";
    compositor.composite(c_partials, result);
    std::cout<<"done COMP\n";
    dray::Framebuffer fb = detail::partials_to_framebuffer(result, width, height);
    fb.composite_background();
    fb.save(image_name);
    //vtkh::Compositor compositor;
    //compositor.SetCompositeMode(vtkh::Compositor::Z_BUFFER_SURFACE);

    //for(int i = 0; i < num_domains; ++i)
    //{
    //  const float * cbuffer =
    //    reinterpret_cast<const float*>(color_buffers[i].get_host_ptr_const());
    //  compositor.AddImage(cbuffer,
    //                      depth_buffers[i].get_host_ptr_const(),
    //                      width,
    //                      height);
    //}
    //vtkh::Image result = compositor.Composite();

    //if(vtkh::GetMPIRank() == 0)
    //{
    //  PNGEncoder encoder;
    //  encoder.Encode(&result.m_pixels[0], width, height);
    //  encoder.Save(image_name + ".png");
    //}
    //std::cout<<"xxxBN\n";
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





