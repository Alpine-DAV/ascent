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
#include <dray/io/blueprint_reader.hpp>

#include <vtkh/vtkh.hpp>
#include <vtkh/rendering/Compositor.hpp>

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





