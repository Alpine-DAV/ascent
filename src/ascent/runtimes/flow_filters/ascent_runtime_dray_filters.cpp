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
#include <ascent_metadata.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_resources.hpp>
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
#include <dray/data_model/data_set.hpp>
#include <dray/filters/mesh_boundary.hpp>

#include <dray/data_model/collection.hpp>

#include <dray/filters/reflect.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/volume_balance.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/exports.hpp>
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
bool is_same(const dray::AABB<3> &b1, const dray::AABB<3> &b2)
{
  return
    b1.m_ranges[0].min() == b2.m_ranges[0].min() &&
    b1.m_ranges[1].min() == b2.m_ranges[1].min() &&
    b1.m_ranges[2].min() == b2.m_ranges[2].min() &&
    b1.m_ranges[0].max() == b2.m_ranges[0].max() &&
    b1.m_ranges[1].max() == b2.m_ranges[1].max() &&
    b1.m_ranges[2].max() == b2.m_ranges[2].max();
}

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

void
parse_plane(const conduit::Node &plane, dray::PlaneDetector &plane_d)
{
  typedef dray::Vec<float,3> Vec3f;

  if(plane.has_child("center"))
  {
      conduit::Node n;
      plane["center"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f vec = {{float(coords[0]), float(coords[1]), float(coords[2])}};
      plane_d.m_center = vec;
  }
  else
  {
    ASCENT_ERROR("Plane definition missing 'center'");
  }

  if(plane.has_child("up"))
  {
      conduit::Node n;
      plane["up"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f vec = {{float(coords[0]), float(coords[1]), float(coords[2])}};
      vec.normalize();
      plane_d.m_up = vec;
  }
  else
  {
    ASCENT_ERROR("Plane definition missing 'up'");
  }

  if(plane.has_child("normal"))
  {
      conduit::Node n;
      plane["normal"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      Vec3f vec = {{float(coords[0]), float(coords[1]), float(coords[2])}};
      vec.normalize();
      plane_d.m_view  = vec;
  }
  else
  {
    ASCENT_ERROR("Plane definition missing 'normal'");
  }

  if(plane.has_child("width"))
  {
      plane_d.m_plane_width = plane["width"].to_float64();
  }
  else
  {
    ASCENT_ERROR("Plane definition missing 'width'");
  }

  if(plane.has_child("height"))
  {
      plane_d.m_plane_height = plane["height"].to_float64();
  }
  else
  {
    ASCENT_ERROR("Plane definition missing 'height'");
  }

}

void
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
             std::vector<dray::Camera> &cameras,
             dray::ColorMap &color_map,
             std::string &field_name,
             std::vector<std::string> &image_names)
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

  dray::AABB<3> bounds = dcol->bounds();

  if(params.has_path("camera"))
  {
    const conduit::Node &n_camera = params["camera"];

    dray::Camera camera;
    camera.set_width(width);
    camera.set_height(height);
    camera.reset_to_bounds(bounds);

    detail::parse_camera(n_camera, camera);
    cameras.push_back(camera);
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

    std::string image_name = params["image_prefix"].as_string();
    image_name = expand_family_name(image_name, cycle);
    image_name = output_dir(image_name);
    image_names.push_back(image_name);
  }
}

class DrayCinemaManager
{
protected:
  std::vector<dray::Camera>  m_cameras;
  std::vector<std::string>   m_image_names;
  std::vector<float>         m_phi_values;
  std::vector<float>         m_theta_values;
  std::vector<float>         m_times;
  std::string                m_csv;

  dray::AABB<3>              m_bounds;
  const int                  m_phi;
  const int                  m_theta;
  std::string                m_image_name;
  std::string                m_image_path;
  std::string                m_db_path;
  std::string                m_base_path;
  float                      m_time;
public:
  DrayCinemaManager(dray::AABB<3> bounds,
                    const int phi,
                    const int theta,
                    const std::string image_name,
                    const std::string path)
    : m_bounds(bounds),
      m_phi(phi),
      m_theta(theta),
      m_image_name(image_name),
      m_time(0.f)
  {
    this->create_cinema_cameras(bounds);
    m_csv = "phi,theta,time,FILE\n";

    m_base_path = conduit::utils::join_file_path(path, "cinema_databases");
  }

  DrayCinemaManager()
    : m_phi(0),
      m_theta(0)
  {
    ASCENT_ERROR("Cannot create un-initialized CinemaManger");
  }

  std::string db_path()
  {
       return conduit::utils::join_file_path(m_base_path, m_image_name);
  }

  void set_bounds(dray::AABB<3> &bounds)
  {
    if(!is_same(bounds, m_bounds))
    {
      this->create_cinema_cameras(bounds);
      m_bounds = bounds;
    }
  }

  void add_time_step()
  {
    m_times.push_back(m_time);

    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank == 0 && !conduit::utils::is_directory(m_base_path))
    {
        conduit::utils::create_directory(m_base_path);
    }

    // add a database path
    m_db_path = db_path();

    // note: there is an implicit assumption here that these
    // resources are static and only need to be generated one
    if(rank == 0 && !conduit::utils::is_directory(m_db_path))
    {
        conduit::utils::create_directory(m_db_path);

        // load cinema web resources from compiled in resource tree
        Node cinema_rc;
        ascent::resources::load_compiled_resource_tree("cinema_web",
                                                        cinema_rc);
        if(cinema_rc.dtype().is_empty())
        {
            ASCENT_ERROR("Failed to load compiled resources for cinema_web");
        }

        ascent::resources::expand_resource_tree_to_file_system(cinema_rc,
                                                               m_db_path);
    }

    std::stringstream ss;
    ss<<fixed<<showpoint;
    ss<<std::setprecision(1)<<m_time;
    // add a time step path
    m_image_path = conduit::utils::join_file_path(m_db_path,ss.str());

    if(!conduit::utils::is_directory(m_image_path))
    {
        conduit::utils::create_directory(m_image_path);
    }

    m_time += 1.f;
  }

/*
  void fill_renders(std::vector<vtkh::Render> *renders,
                    const conduit::Node &render_node)
  {
    conduit::Node render_copy = render_node;

    // allow zoom to be ajusted
    conduit::Node zoom;
    if(render_copy.has_path("camera/zoom"))
    {
      zoom = render_copy["camera/zoom"];
    }

    // cinema is controlling the camera so get
    // rid of it
    if(render_copy.has_path("camera"))
    {
      render_copy["camera"].reset();
    }

    std::string tmp_name = "";
    vtkh::Render render = detail::parse_render(render_copy,
                                               m_bounds,
                                               tmp_name);
    const int num_renders = m_image_names.size();

    for(int i = 0; i < num_renders; ++i)
    {
      vtkh::Render tmp = render.Copy();
      std::string image_name = conduit::utils::join_file_path(m_image_path , m_image_names[i]);

      tmp.SetImageName(image_name);
      // we have to make a copy of the camera because
      // zoom is additive for some reason
      vtkm::rendering::Camera camera = m_cameras[i];

      if(!zoom.dtype().is_empty())
      {
        // Allow default zoom to be overridden
        double vtkm_zoom = zoom_to_vtkm_zoom(zoom.to_float64());
        camera.Zoom(vtkm_zoom);
      }

      tmp.SetCamera(camera);
      renders->push_back(tmp);
    }
  }
*/

  std::string get_string(const float value)
  {
    std::stringstream ss;
    ss<<std::fixed<<std::setprecision(1)<<value;
    return ss.str();
  }

  void write_metadata()
  {
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank != 0)
    {
      return;
    }
    conduit::Node meta;
    meta["type"] = "simple";
    meta["version"] = "1.1";
    meta["metadata/type"] = "parametric-image-stack";
    meta["name_pattern"] = "{time}/{phi}_{theta}_" + m_image_name + ".png";

    conduit::Node times;
    times["default"] = get_string(m_times[0]);
    times["label"] = "time";
    times["type"] = "range";
    // we have to make sure that this maps to a json array
    const int t_size = m_times.size();
    for(int i = 0; i < t_size; ++i)
    {
      times["values"].append().set(get_string(m_times[i]));
    }

    meta["arguments/time"] = times;

    conduit::Node phis;
    phis["default"] = get_string(m_phi_values[0]);
    phis["label"] = "phi";
    phis["type"] = "range";
    const int phi_size = m_phi_values.size();
    for(int i = 0; i < phi_size; ++i)
    {
      phis["values"].append().set(get_string(m_phi_values[i]));
    }

    meta["arguments/phi"] = phis;

    conduit::Node thetas;
    thetas["default"] = get_string(m_theta_values[0]);
    thetas["label"] = "theta";
    thetas["type"] = "range";
    const int theta_size = m_theta_values.size();
    for(int i = 0; i < theta_size; ++i)
    {
      thetas["values"].append().set(get_string(m_theta_values[i]));
    }

    meta["arguments/theta"] = thetas;
    meta.save(m_db_path + "/info.json","json");

    // also generate info.js, a simple javascript variant of
    // info.json that our index.html reads directly to
    // avoid ajax

    std::ofstream out_js(m_db_path + "/info.js");
    out_js<<"var info =";
    meta.to_json_stream(out_js);
    out_js.close();

    //append current data to our csv file
    std::stringstream csv;

    csv<<m_csv;
    std::string current_time = get_string(m_times[t_size - 1]);
    for(int p = 0; p < phi_size; ++p)
    {
      std::string phi = get_string(m_phi_values[p]);
      for(int t = 0; t < theta_size; ++t)
      {
        std::string theta = get_string(m_theta_values[t]);
        csv<<phi<<",";
        csv<<theta<<",";
        csv<<current_time<<",";
        csv<<current_time<<"/"<<phi<<"_"<<theta<<"_"<<m_image_name<<".png\n";
      }
    }

    m_csv = csv.str();
    std::ofstream out(m_db_path + "/data.csv");
    out<<m_csv;
    out.close();

  }

private:
  void create_cinema_cameras(dray::AABB<3> bounds)
  {
    m_cameras.clear();
    m_image_names.clear();
    using Vec3f = dray::Vec<float,3>;
    Vec3f center = bounds.center();
    Vec3f totalExtent;
    totalExtent[0] = float(bounds.m_ranges[0].length());
    totalExtent[1] = float(bounds.m_ranges[1].length());
    totalExtent[2] = float(bounds.m_ranges[2].length());

    float radius = totalExtent.magnitude() * 2.5 / 2.0;

    const double pi = 3.141592653589793;
    double phi_inc = 360.0 / double(m_phi);
    double theta_inc = 180.0 / double(m_theta);
    for(int p = 0; p < m_phi; ++p)
    {
      float phi  =  -180.f + phi_inc * p;
      m_phi_values.push_back(phi);

      for(int t = 0; t < m_theta; ++t)
      {
        float theta = theta_inc * t;
        if (p == 0)
        {
          m_theta_values.push_back(theta);
        }

        const int i = p * m_theta + t;

        dray::Camera camera;
        camera.reset_to_bounds(bounds);

        //
        //  spherical coords start (r=1, theta = 0, phi = 0)
        //  (x = 0, y = 0, z = 1)
        //

        Vec3f pos = {{0.f,0.f,1.f}};
        Vec3f up = {{0.f,1.f,0.f}};

        dray::Matrix<float,4,4> phi_rot;
        dray::Matrix<float,4,4> theta_rot;
        dray::Matrix<float,4,4> rot;

        phi_rot = dray::rotate_z(phi);
        theta_rot = dray::rotate_x(theta);
        rot = phi_rot * theta_rot;

        up = dray::transform_vector(rot, up);
        up.normalize();

        pos = dray::transform_point(rot, pos);
        pos = pos * radius + center;

        camera.set_up(up);
        camera.set_look_at(center);
        camera.set_pos(pos);
        //camera.Zoom(0.2f);

        std::stringstream ss;
        ss<<get_string(phi)<<"_"<<get_string(theta)<<"_";

        m_image_names.push_back(ss.str() + m_image_name);
        m_cameras.push_back(camera);

      } // theta
    } // phi
  }

}; // DrayCinemaManager

class DrayCinemaDatabases
{
private:
  static std::map<std::string, DrayCinemaManager> m_databases;
public:

  static bool db_exists(std::string db_name)
  {
    auto it = m_databases.find(db_name);
    return it != m_databases.end();
  }

  static void create_db(dray::AABB<3> &bounds,
                        const int phi,
                        const int theta,
                        std::string db_name,
                        std::string path)
  {
    if(db_exists(db_name))
    {
      ASCENT_ERROR("Creation failed: dray cinema database already exists");
    }

    m_databases.emplace(std::make_pair(db_name, DrayCinemaManager(bounds, phi, theta, db_name, path)));
  }

  static DrayCinemaManager& get_db(std::string db_name)
  {
    if(!db_exists(db_name))
    {
      ASCENT_ERROR("DRAY Cinema db '"<<db_name<<"' does not exist.");
    }

    return m_databases[db_name];
  }
}; // DrayCinemaDatabases

std::map<std::string, DrayCinemaManager> DrayCinemaDatabases::m_databases;

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
    if(!d_input->is_valid())
    {
      return;
    }

    dray::Collection *dcol = d_input->as_dray_collection().get();
    int comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    comm_id = flow::Workspace::default_mpi_comm();
#endif
    bool is_3d = dcol->topo_dims() == 3;

    dray::Collection faces = detail::boundary(*dcol);

    std::vector<dray::Camera> cameras;
    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::vector<std::string> image_names;
    conduit::Node meta = Metadata::n_metadata;

    detail::parse_params(params(),
                         &faces,
                         &meta,
                         cameras,
                         color_map,
                         field_name,
                         image_names);

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

    const int num_images = cameras.size();
    for(int i = 0; i < num_images; ++i)
    {
      dray::Camera &camera = cameras[i];
      dray::Framebuffer fb = renderer.render(camera);

      if(dray::dray::mpi_rank() == 0)
      {
        fb.composite_background();
        fb.save(image_names[i]);
      }
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
    if(!d_input->is_valid())
    {
      return;
    }

    dray::Collection *dcol = d_input->as_dray_collection().get();

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


    const int num_images = cameras.size();
    for(int i = 0; i < num_images; ++i)
    {
      dray::Camera &camera = cameras[i];
      dray::Framebuffer fb = renderer.render(camera);

      if(dray::dray::mpi_rank() == 0)
      {
        fb.composite_background();
        fb.save(image_names[i]);
      }
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
    if(!d_input->is_valid())
    {
      return;
    }

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

        // We should have at least one camera so just use that for load
        // balancing. Matt: this could be bad but right now the only case
        // we use multiple images for is ciname
        dataset = balancer.execute(dataset, cameras[0], samples);

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

    const int num_images = cameras.size();
    for(int i = 0; i < num_images; ++i)
    {
      dray::Camera &camera = cameras[i];
      dray::Framebuffer fb = renderer.render(camera);

      if(dray::dray::mpi_rank() == 0)
      {
        fb.composite_background();
        fb.save(image_names[i]);
      }
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
    if(!d_input->is_valid())
    {
      set_output<DataObject>(d_input);
      return;
    }

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

    for(int i = 0; i < dcol->local_size(); ++i)
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
    ignore_paths.push_back("plane");
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
    if(!d_input->is_valid())
    {
      set_output<DataObject>(d_input);
      return;
    }

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection faces = detail::boundary(*dcol);

    std::string image_name;

    conduit::Node meta = Metadata::n_metadata;
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

    dray::PlaneDetector plane;
    bool use_plane = false;
    if(params().has_path("plane"))
    {
      use_plane = true;
      detail::parse_plane(params()["plane"], plane);
      plane.m_x_res = width;
      plane.m_y_res = height;
    }
    else if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      detail::parse_camera(n_camera, camera);
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
    dray::ScalarBuffer sb;
    if(use_plane)
    {
      sb = renderer.render(plane);
    }
    else
    {
      sb = renderer.render(camera);
    }

    conduit::Node *output = new conduit::Node();
    if(dray::dray::mpi_rank() == 0)
    {
      conduit::Node &dom = output->append();

      sb.to_node(dom);

      dom["state/domain_id"] = 0;

      int cycle = 0;

      if(meta.has_path("cycle"))
      {
        cycle = meta["cycle"].to_int32();
      }
      dom["state/cycle"] = cycle;
      if(meta.has_path("time"))
      {
        dom["state/time"] =  meta["time"].to_float64();
      }

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
    if(!d_input->is_valid())
    {
      set_output<DataObject>(d_input);
      return;
    }

    dray::Collection *dcol = d_input->as_dray_collection().get();

    dray::Collection faces = detail::boundary(*dcol);

    std::vector<dray::Camera> cameras;
    dray::ColorMap color_map("cool2warm");
    std::string field_name;
    std::vector<std::string> image_names;
    conduit::Node meta = Metadata::n_metadata;

    detail::parse_params(params(),
                         &faces,
                         &meta,
                         cameras,
                         color_map,
                         field_name,
                         image_names);

    if(cameras.size() != 1)
    {
      ASCENT_ERROR("DrayProjectColors: only one camera is supported (no cinema)");
    }

    dray::Camera camera = cameras[0];

    const int num_domains = faces.local_size();

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
DRayVectorComponent::DRayVectorComponent()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayVectorComponent::~DRayVectorComponent()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayVectorComponent::declare_interface(Node &i)
{
    i["type_name"]   = "dray_vector_component";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DRayVectorComponent::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("component",params, info, true);
    res &= check_string("output_name",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("component");
    valid_paths.push_back("output_name");

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
DRayVectorComponent::execute()
{

    if(!input(0).check_type<DataObject>())
    {
      ASCENT_ERROR("dray_vector_component input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    dray::Collection *dcol = data_object->as_dray_collection().get();

    std::string field_name = params()["field"].as_string();
    // not really doing invalid results for dray atm
    //if(!collection->has_field(field_name))
    //{
    //  // this creates a data object with an invalid soource
    //  set_output<DataObject>(new DataObject());
    //  return;
    //}
    int component = params()["component"].to_int32();
    std::string res_name = params()["output_name"].as_string();

    dray::VectorComponent comp;

    comp.field(field_name);
    comp.component(component);
    comp.output_name(res_name);

    dray::Collection comp_output = comp.execute(*dcol);
    dray::Collection *output_ptr = new dray::Collection();
    *output_ptr = comp_output;

    DataObject *res =  new DataObject(output_ptr);
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





