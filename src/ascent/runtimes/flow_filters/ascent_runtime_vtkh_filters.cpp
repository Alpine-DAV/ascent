//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory //
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
/// file: ascent_runtime_vtkh_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_vtkh_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/filters/Statistics.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/filters/HistSampling.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <stdio.h>

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

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------
namespace detail
{
std::string
check_color_table_surprises(const conduit::Node &color_table)
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
check_renders_surprises(const conduit::Node &renders_node)
{
  std::string surprises;
  const int num_renders = renders_node.number_of_children();
  // render paths
  std::vector<std::string> r_valid_paths;
  r_valid_paths.push_back("image_name");
  r_valid_paths.push_back("image_prefix");
  r_valid_paths.push_back("image_width");
  r_valid_paths.push_back("image_height");
  r_valid_paths.push_back("scene_bounds");
  r_valid_paths.push_back("camera/look_at");
  r_valid_paths.push_back("camera/position");
  r_valid_paths.push_back("camera/up");
  r_valid_paths.push_back("camera/fov");
  r_valid_paths.push_back("camera/xpan");
  r_valid_paths.push_back("camera/ypan");
  r_valid_paths.push_back("camera/zoom");
  r_valid_paths.push_back("camera/near_plane");
  r_valid_paths.push_back("camera/far_plane");
  r_valid_paths.push_back("camera/azimuth");
  r_valid_paths.push_back("camera/elevation");
  r_valid_paths.push_back("type");
  r_valid_paths.push_back("phi");
  r_valid_paths.push_back("theta");
  r_valid_paths.push_back("db_name");
  r_valid_paths.push_back("render_bg");
  r_valid_paths.push_back("annotations");
  r_valid_paths.push_back("output_path");
  r_valid_paths.push_back("fg_color");
  r_valid_paths.push_back("bg_color");

  for(int i = 0; i < num_renders; ++i)
  {
    const conduit::Node &render_node = renders_node.child(i);
    surprises += surprise_check(r_valid_paths, render_node);
  }
  return surprises;
}
// A simple container to create registry entries for
// renderer and the data set it renders. Without this,
// pipeline results (data sets) would be deleted before
// the Scene can be executed.
//
class RendererContainer
{
protected:
  std::string m_key;
  flow::Registry *m_registry;
  std::string m_data_set_key;
  RendererContainer() {};
public:
  RendererContainer(std::string key,
                    flow::Registry *r,
                    vtkh::Renderer *renderer)
    : m_key(key),
      m_registry(r)
  {
    m_data_set_key = m_key + "_dset";
    m_registry->add<vtkh::Renderer>(m_key,renderer,1);
    m_registry->add<vtkh::DataSet>(m_data_set_key, renderer->GetInput(),1);
  }

  vtkh::Renderer *
  Fetch()
  {
    return m_registry->fetch<vtkh::Renderer>(m_key);
  }

  ~RendererContainer()
  {
    m_registry->consume(m_key);
    m_registry->consume(m_data_set_key);
  }
};


class AscentScene
{
protected:
  int m_renderer_count;
  flow::Registry *m_registry;
  AscentScene() {};
public:

  AscentScene(flow::Registry *r)
    : m_registry(r),
      m_renderer_count(0)
  {}

  ~AscentScene()
  {}

  void AddRenderer(RendererContainer *container)
  {
    ostringstream oss;
    oss << "key_" << m_renderer_count;
    m_registry->add<RendererContainer>(oss.str(),container,1);

    m_renderer_count++;
  }

  void Execute(std::vector<vtkh::Render> &renders)
  {
    vtkh::Scene scene;
    for(int i = 0; i < m_renderer_count; i++)
    {
      ostringstream oss;
      oss << "key_" << i;
      vtkh::Renderer * r = m_registry->fetch<RendererContainer>(oss.str())->Fetch();
      scene.AddRenderer(r);
    }

    size_t num_renders = renders.size();
    for(size_t i = 0; i < num_renders; ++i)
    {
      scene.AddRender(renders[i]);
    }

    scene.Render();

    for(int i=0; i < m_renderer_count; i++)
    {
        ostringstream oss;
        oss << "key_" << i;
        m_registry->consume(oss.str());
    }
  }
}; // Ascent Scene

//-----------------------------------------------------------------------------

vtkh::Render parse_render(const conduit::Node &render_node,
                          vtkm::Bounds &bounds,
                          const std::vector<vtkm::Id> &domain_ids,
                          const std::string &image_name)
{
  int image_width;
  int image_height;

  parse_image_dims(render_node, image_width, image_height);

  //
  // for now, all the canvases we support are the same
  // so passing MakeRender a RayTracer is ok
  //
  vtkh::Render render = vtkh::MakeRender(image_width,
                                         image_height,
                                         bounds,
                                         domain_ids,
                                         image_name);
  //
  // render create a default camera. Now get it and check for
  // values that override the default view
  //
  if(render_node.has_path("camera"))
  {
    vtkm::rendering::Camera camera = render.GetCamera();
    parse_camera(render_node["camera"], camera);
    render.SetCamera(camera);
  }

  if(render_node.has_path("annotations"))
  {
    if(!render_node["annotations"].dtype().is_string())
    {
      ASCENT_ERROR("render/annotations node must be a string value");
    }
    const std::string annot = render_node["annotations"].as_string();
    // default is always render annotations
    if(annot == "false")
    {
      render.DoRenderAnnotations(false);
    }
  }

  if(render_node.has_path("render_bg"))
  {
    if(!render_node["render_bg"].dtype().is_string())
    {
      ASCENT_ERROR("render/render_bg node must be a string value");
    }
    const std::string render_bg = render_node["render_bg"].as_string();
    // default is always render the background
    // off will make the background transparent
    if(render_bg == "false")
    {
      render.DoRenderBackground(false);
    }
  }

  if(render_node.has_path("bg_color"))
  {
    if(!render_node["bg_color"].dtype().is_number() ||
       render_node["bg_color"].dtype().number_of_elements() != 3)
    {
      ASCENT_ERROR("render/bg_color node must be an array of 3 values");
    }
    conduit::Node n;
    render_node["bg_color"].to_float32_array(n);
    const float32 *color = n.as_float32_ptr();
    float32 color4f[4];
    color4f[0] = color[0];
    color4f[1] = color[1];
    color4f[2] = color[2];
    color4f[3] = 1.f;
    render.SetBackgroundColor(color4f);
  }

  if(render_node.has_path("fg_color"))
  {
    if(!render_node["fg_color"].dtype().is_number() ||
       render_node["fg_color"].dtype().number_of_elements() != 3)
    {
      ASCENT_ERROR("render/fg_color node must be an array of 3 values");
    }
    conduit::Node n;
    render_node["fg_color"].to_float32_array(n);
    const float32 *color = n.as_float32_ptr();
    float32 color4f[4];
    color4f[0] = color[0];
    color4f[1] = color[1];
    color4f[2] = color[2];
    color4f[3] = 1.f;
    render.SetForegroundColor(color4f);
  }


  return render;
}

class CinemaManager
{
protected:
  std::vector<vtkm::rendering::Camera> m_cameras;
  std::vector<std::string>             m_image_names;
  std::vector<float>                   m_phi_values;
  std::vector<float>                   m_theta_values;
  std::vector<float>                   m_times;
  std::string                          m_csv;

  vtkm::Bounds                         m_bounds;
  const int                            m_phi;
  const int                            m_theta;
  std::string                          m_image_name;
  std::string                          m_image_path;
  std::string                          m_db_path;
  std::string                          m_base_path;
  float                                m_time;
public:
  CinemaManager(vtkm::Bounds bounds,
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

  CinemaManager()
    : m_phi(0),
      m_theta(0)
  {
    ASCENT_ERROR("Cannot create un-initialized CinemaManger");
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
    m_db_path = conduit::utils::join_file_path(m_base_path, m_image_name);

    if(rank == 0 && !conduit::utils::is_directory(m_db_path))
    {
        conduit::utils::create_directory(m_db_path);
        // copy over cinema web resources
        std::string cinema_root = conduit::utils::join_file_path(ASCENT_WEB_CLIENT_ROOT,
                                                                 "cinema");
        ascent::copy_directory(cinema_root, m_db_path);
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

  void fill_renders(std::vector<vtkh::Render> *renders,
                    const std::vector<vtkm::Id> &domain_ids,
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
                                               domain_ids,
                                               tmp_name);
    const int num_renders = m_image_names.size();

    for(int i = 0; i < num_renders; ++i)
    {
      std::string image_name = conduit::utils::join_file_path(m_image_path , m_image_names[i]);

      render.SetImageName(image_name);

      if(!zoom.dtype().is_empty())
      {
        // Allow default zoom to be overridden
        m_cameras[i].Zoom(zoom.to_float32());
      }

      render.SetCamera(m_cameras[i]);
      renders->push_back(render);
    }
  }

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
  void create_cinema_cameras(vtkm::Bounds bounds)
  {
    using vtkmVec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkmVec3f center = bounds.Center();
    vtkm::Vec<vtkm::Float32,3> totalExtent;
    totalExtent[0] = vtkm::Float32(bounds.X.Length());
    totalExtent[1] = vtkm::Float32(bounds.Y.Length());
    totalExtent[2] = vtkm::Float32(bounds.Z.Length());

    vtkm::Float32 radius = vtkm::Magnitude(totalExtent) * 2.5 / 2.0;

    const double pi = 3.141592653589793;
    double phi_inc = 360.0 / double(m_phi);
    double theta_inc = 180.0 / double(m_theta);
    for(int p = 0; p < m_phi; ++p)
    {
      for(int t = 0; t < m_theta; ++t)
      {
        float phi  =  -180.f + phi_inc * p;
        float theta = -90.f + theta_inc * t;

        const int i = p * m_theta + t;

        vtkm::rendering::Camera camera;
        camera.ResetToBounds(bounds);

        //
        //  spherical coords start (r=1, theta = 0, phi = 0)
        //  (x = 0, y = 0, z = 1)
        //

        vtkmVec3f pos(0.f,0.f,1.f);
        vtkmVec3f up(0.f,1.f,0.f);

        vtkm::Matrix<vtkm::Float32,4,4> phi_rot;
        vtkm::Matrix<vtkm::Float32,4,4> theta_rot;
        vtkm::Matrix<vtkm::Float32,4,4> rot;

        phi_rot = vtkm::Transform3DRotateZ(phi);
        theta_rot = vtkm::Transform3DRotateX(theta);
        rot = vtkm::MatrixMultiply(phi_rot, theta_rot);

        up = vtkm::Transform3DVector(rot, up);
        vtkm::Normalize(up);

        pos = vtkm::Transform3DPoint(rot, pos);
        pos = pos * radius + center;

        camera.SetViewUp(up);
        camera.SetLookAt(center);
        camera.SetPosition(pos);
        camera.Zoom(0.2f);

        std::stringstream ss;
        ss<<get_string(phi)<<"_"<<get_string(theta)<<"_";

        m_image_names.push_back(ss.str() + m_image_name);
        m_cameras.push_back(camera);

      } // theta
    } // phi

    for(int p = 0; p < m_phi; ++p)
    {
      float phi  =  -180.f + phi_inc * p;
      m_phi_values.push_back(phi);
    }

    for(int t = 0; t < m_theta; ++t)
    {
      float theta = -90.f + theta_inc * t;
      m_theta_values.push_back(theta);
    }

  }

}; // CinemaManager

class CinemaDatabases
{
private:
  static std::map<std::string, CinemaManager> m_databases;
public:

  static bool db_exists(std::string db_name)
  {
    auto it = m_databases.find(db_name);
    return it != m_databases.end();
  }

  static void create_db(vtkm::Bounds bounds,
                        const int phi,
                        const int theta,
                        std::string db_name,
                        std::string path)
  {
    if(db_exists(db_name))
    {
      ASCENT_ERROR("Creation failed: cinema database already exists");
    }

    m_databases.emplace(std::make_pair(db_name, CinemaManager(bounds, phi, theta, db_name, path)));
  }

  static CinemaManager& get_db(std::string db_name)
  {
    if(!db_exists(db_name))
    {
      ASCENT_ERROR("Cinema db '"<<db_name<<"' does not exist.");
    }

    return m_databases[db_name];
  }
};

std::map<std::string, CinemaManager> CinemaDatabases::m_databases;

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end namespace detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
EnsureVTKH::EnsureVTKH()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
EnsureVTKH::~EnsureVTKH()
{
// empty
}

//-----------------------------------------------------------------------------
void
EnsureVTKH::declare_interface(Node &i)
{
    i["type_name"]   = "ensure_vtkh";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void
EnsureVTKH::execute()
{
    bool zero_copy = false;
    if(params().has_path("zero_copy"))
    {
      if(params()["zero_copy"].as_string() == "true")
      {
        zero_copy = true;
      }
    }

    if(input(0).check_type<Node>())
    {
        // convert from blueprint to vtk-h
        const Node *n_input = input<Node>(0);

        vtkh::DataSet *res = nullptr;;
        res = VTKHDataAdapter::BlueprintToVTKHDataSet(*n_input, zero_copy);

        set_output<vtkh::DataSet>(res);
    }
    else if(input(0).check_type<vtkm::cont::DataSet>())
    {
        // wrap our vtk-m dataset in vtk-h
        vtkh::DataSet *res = VTKHDataAdapter::VTKmDataSetToVTKHDataSet(input<vtkm::cont::DataSet>(0));
        set_output<vtkh::DataSet>(res);
    }
    else if(input(0).check_type<vtkh::DataSet>())
    {
        // our data is already vtkh, pass though
        set_output(input(0));
    }
    else
    {
        ASCENT_ERROR("ensure_vtkh input must be a mesh blueprint "
                     "conforming conduit::Node, a vtk-m dataset, or vtk-h dataset");
    }
}


//-----------------------------------------------------------------------------
VTKHMarchingCubes::VTKHMarchingCubes()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHMarchingCubes::~VTKHMarchingCubes()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHMarchingCubes::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_marchingcubes";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHMarchingCubes::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    bool has_values = check_numeric("iso_values",params, info, false);
    bool has_levels = check_numeric("levels",params, info, false);

    if(!has_values && !has_levels)
    {
        info["errors"].append() = "Missing required numeric parameter. Contour must"
                                  " specify 'iso_values' or 'levels'.";
        res = false;
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("levels");
    valid_paths.push_back("iso_values");
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
VTKHMarchingCubes::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_marchingcubes input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::MarchingCubes marcher;

    marcher.SetInput(data);
    marcher.SetField(field_name);

    if(params().has_path("iso_values"))
    {
      const Node &n_iso_vals = params()["iso_values"];

      // convert to contig doubles
      Node n_iso_vals_dbls;
      n_iso_vals.to_float64_array(n_iso_vals_dbls);

      marcher.SetIsoValues(n_iso_vals_dbls.as_double_ptr(),
                           n_iso_vals_dbls.dtype().number_of_elements());
    }
    else
    {
      marcher.SetLevels(params()["levels"].to_int32());
    }

    marcher.Update();

    vtkh::DataSet *iso_output = marcher.GetOutput();
    set_output<vtkh::DataSet>(iso_output);
}

//-----------------------------------------------------------------------------
VTKHVectorMagnitude::VTKHVectorMagnitude()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVectorMagnitude::~VTKHVectorMagnitude()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVectorMagnitude::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_vector_magnitude";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHVectorMagnitude::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res = check_string("output_name",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
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
VTKHVectorMagnitude::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_vector_magnitude input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::VectorMagnitude mag;

    mag.SetInput(data);
    mag.SetField(field_name);
    if(params().has_path("output_name"))
    {
      std::string output_name = params()["output_name"].as_string();
      mag.SetResultName(output_name);
    }

    mag.Update();

    vtkh::DataSet *mag_output = mag.GetOutput();
    set_output<vtkh::DataSet>(mag_output);
}

//-----------------------------------------------------------------------------
VTKH3Slice::VTKH3Slice()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKH3Slice::~VTKH3Slice()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKH3Slice::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_3slice";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKH3Slice::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = true;
    // this can have no parameters
    if(params.number_of_children() != 0)
    {
      res &= check_numeric("x_offset",params, info, false);
      res &= check_numeric("y_offset",params, info, false);
      res &= check_numeric("z_offset",params, info, false);

      std::vector<std::string> valid_paths;
      valid_paths.push_back("x_offset");
      valid_paths.push_back("y_offset");
      valid_paths.push_back("z_offset");
      std::string surprises = surprise_check(valid_paths, params);

      if(surprises != "")
      {
        res = false;
        info["errors"].append() = surprises;
      }
    }
    return res;
}

//-----------------------------------------------------------------------------
void
VTKH3Slice::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_3slice input must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Slice slicer;

    slicer.SetInput(data);

    using Vec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkm::Bounds bounds = data->GetGlobalBounds();
    Vec3f center = bounds.Center();
    Vec3f x_point = center;
    Vec3f y_point = center;
    Vec3f z_point = center;

    //
    // We look for offsets for each slice plane.
    // Offset values are between -1 and 1 where -1 pushes the plane
    // to the min extent on the bounds and 1 pushes the plane to
    // the max extent
    //

    const float eps = 1e-5; // ensure that the slice is always inside the data set
    if(params().has_path("x_offset"))
    {
      float offset = params()["x_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      x_point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);
    }

    if(params().has_path("y_offset"))
    {
      float offset = params()["y_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      y_point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);
    }

    if(params().has_path("z_offset"))
    {
      float offset = params()["z_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      z_point[2] = bounds.Z.Min + t * (bounds.Z.Max - bounds.Z.Min);
    }

    Vec3f x_normal(1.f, 0.f, 0.f);
    Vec3f y_normal(0.f, 1.f, 0.f);
    Vec3f z_normal(0.f, 0.f, 1.f);


    slicer.AddPlane(x_point, x_normal);
    slicer.AddPlane(y_point, y_normal);
    slicer.AddPlane(z_point, z_normal);
    slicer.Update();

    vtkh::DataSet *slice_output = slicer.GetOutput();

    set_output<vtkh::DataSet>(slice_output);
}

//-----------------------------------------------------------------------------
VTKHSlice::VTKHSlice()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHSlice::~VTKHSlice()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHSlice::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_slice";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHSlice::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = true;


    if(params.has_path("point/x_offset") && params.has_path("point/x"))
    {
      info["errors"]
        .append() = "Cannot specify the plane point as both an offset and explicit point";
      res = false;
    }

    if(params.has_path("point/x"))
    {
      res &= check_numeric("point/x",params, info, true);
      res = check_numeric("point/y",params, info, true) && res;
      res = check_numeric("point/z",params, info, true) && res;
    }
    else if(params.has_path("point/x_offset"))
    {
      res &= check_numeric("point/x_offset",params, info, true);
      res = check_numeric("point/y_offset",params, info, true) && res;
      res = check_numeric("point/z_offset",params, info, true) && res;
    }
    else
    {
      info["errors"]
        .append() = "Slice must specify a point for the plane.";
      res = false;
    }


    res = check_numeric("normal/x",params, info, true) && res;
    res = check_numeric("normal/y",params, info, true) && res;
    res = check_numeric("normal/z",params, info, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("point/x");
    valid_paths.push_back("point/y");
    valid_paths.push_back("point/z");
    valid_paths.push_back("point/x_offset");
    valid_paths.push_back("point/y_offset");
    valid_paths.push_back("point/z_offset");
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
VTKHSlice::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_slice input must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Slice slicer;

    slicer.SetInput(data);

    const Node &n_point = params()["point"];
    const Node &n_normal = params()["normal"];

    using Vec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkm::Bounds bounds = data->GetGlobalBounds();
    Vec3f point;

    const float eps = 1e-5; // ensure that the slice is always inside the data set


    if(n_point.has_path("x_offset"))
    {
      float offset = n_point["x_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);

      offset = n_point["y_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);

      offset = n_point["z_offset"].to_float32();
      std::max(-1.f, std::min(1.f, offset));
      t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      point[2] = bounds.Z.Min + t * (bounds.Z.Max - bounds.Z.Min);
    }
    else
    {
      point[0] = n_point["x"].to_float32();
      point[1] = n_point["y"].to_float32();
      point[2] = n_point["z"].to_float32();
    }

    vtkm::Vec<vtkm::Float32,3> v_normal(n_normal["x"].to_float32(),
                                        n_normal["y"].to_float32(),
                                        n_normal["z"].to_float32());

    slicer.AddPlane(point, v_normal);
    slicer.Update();

    vtkh::DataSet *slice_output = slicer.GetOutput();

    set_output<vtkh::DataSet>(slice_output);
}

//-----------------------------------------------------------------------------
VTKHGhostStripper::VTKHGhostStripper()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHGhostStripper::~VTKHGhostStripper()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHGhostStripper::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_ghost_stripper";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHGhostStripper::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    res = check_numeric("min_value",params, info, true) && res;
    res = check_numeric("max_value",params, info, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
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
VTKHGhostStripper::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHGhostStripper input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);

    // Check to see of the ghost field even exists
    bool do_strip = data->GlobalFieldExists(field_name);

    if(do_strip)
    {
      vtkh::GhostStripper stripper;

      stripper.SetInput(data);
      stripper.SetField(field_name);

      const Node &n_min_val = params()["min_value"];
      const Node &n_max_val = params()["max_value"];

      int min_val = n_min_val.to_int32();
      int max_val = n_max_val.to_int32();

      stripper.SetMaxValue(max_val);
      stripper.SetMinValue(min_val);

      stripper.Update();

      vtkh::DataSet *stripper_output = stripper.GetOutput();

      set_output<vtkh::DataSet>(stripper_output);
    }
    else
    {
      set_output<vtkh::DataSet>(data);
    }
}

//-----------------------------------------------------------------------------
VTKHThreshold::VTKHThreshold()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHThreshold::~VTKHThreshold()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHThreshold::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_threshold";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHThreshold::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    res = check_numeric("min_value",params, info, true) && res;
    res = check_numeric("max_value",params, info, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
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
VTKHThreshold::execute()
{


    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHThresholds input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Threshold thresher;

    thresher.SetInput(data);
    thresher.SetField(field_name);

    const Node &n_min_val = params()["min_value"];
    const Node &n_max_val = params()["max_value"];

    // convert to contig doubles
    double min_val = n_min_val.to_float64();
    double max_val = n_max_val.to_float64();
    thresher.SetUpperThreshold(max_val);
    thresher.SetLowerThreshold(min_val);

    thresher.Update();

    vtkh::DataSet *thresh_output = thresher.GetOutput();

    set_output<vtkh::DataSet>(thresh_output);
}

//-----------------------------------------------------------------------------
DefaultRender::DefaultRender()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DefaultRender::~DefaultRender()
{
// empty
}

//-----------------------------------------------------------------------------
void
DefaultRender::declare_interface(Node &i)
{
    i["type_name"] = "default_render";
    i["port_names"].append() = "a";
    i["port_names"].append() = "b";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DefaultRender::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("image_name",params, info, false);
    res &= check_string("image_prefix",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("image_prefix");
    valid_paths.push_back("image_name");

    std::vector<std::string> ignore_paths;
    ignore_paths.push_back("renders");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);


    // parse render surprises
    if(params.has_path("renders"))
    {
      const conduit::Node &renders_node = params["renders"];
      surprises += detail::check_renders_surprises(renders_node);
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
DefaultRender::execute()
{


    if(!input(0).check_type<vtkm::Bounds>())
    {
      ASCENT_ERROR("'a' input must be a vktm::Bounds * instance");
    }

    if(!input(1).check_type<std::set<vtkm::Id> >())
    {
        ASCENT_ERROR("'b' must be a std::set<vtkm::Id> * instance");
    }

    vtkm::Bounds *bounds = input<vtkm::Bounds>(0);
    std::set<vtkm::Id> *domain_ids = input<std::set<vtkm::Id>>(1);
    std::vector<vtkm::Id> v_domain_ids(domain_ids->size());
    std::copy(domain_ids->begin(), domain_ids->end(), v_domain_ids.begin());

    std::vector<vtkh::Render> *renders = new std::vector<vtkh::Render>();

    Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    int cycle = 0;

    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    if(params().has_path("renders"))
    {
      const conduit::Node renders_node = params()["renders"];
      const int num_renders = renders_node.number_of_children();

      for(int i = 0; i < num_renders; ++i)
      {
        const conduit::Node render_node = renders_node.child(i);
        std::string image_name;

        bool is_cinema = false;

        if(render_node.has_path("type"))
        {
          if(render_node["type"].as_string() == "cinema")
          {
            is_cinema = true;
          }
        }

        if(is_cinema)
        {
          if(!render_node.has_path("phi") || !render_node.has_path("theta"))
          {
            ASCENT_ERROR("Cinema must have 'phi' and 'theta'");
          }
          int phi = render_node["phi"].to_int32();
          int theta = render_node["theta"].to_int32();

          if(!render_node.has_path("db_name"))
          {
            ASCENT_ERROR("Cinema must specify a 'db_name'");
          }

          std::string output_path = "";

          if(render_node.has_path("output_path"))
          {
            output_path = render_node["output_path"].as_string();
          }

          if(!render_node.has_path("db_name"))
          {
            ASCENT_ERROR("Cinema must specify a 'db_name'");
          }
          std::string db_name = render_node["db_name"].as_string();
          bool exists = detail::CinemaDatabases::db_exists(db_name);
          if(!exists)
          {
            detail::CinemaDatabases::create_db(*bounds,phi,theta, db_name, output_path);
          }
          detail::CinemaManager &manager = detail::CinemaDatabases::get_db(db_name);

          int image_width;
          int image_height;
          parse_image_dims(render_node, image_width, image_height);

          manager.add_time_step();
          manager.fill_renders(renders, v_domain_ids, render_node);
          manager.write_metadata();
        }

        else
        {
          // this render has a unique name
          if(render_node.has_path("image_name"))
          {
            image_name = render_node["image_name"].as_string();
          }
          else if(render_node.has_path("image_prefix"))
          {
            std::stringstream ss;
            ss<<expand_family_name(render_node["image_prefix"].as_string(), cycle);
            image_name = ss.str();
          }
          else
          {
            ASCENT_ERROR("Render must have either a 'image_name' or 'image_prefix' parameter");
          }

          vtkh::Render render = detail::parse_render(render_node,
                                                     *bounds,
                                                     v_domain_ids,
                                                     image_name);
          renders->push_back(render);
        }
      }
    }
    else
    {
      // This is the path for the default render attached directly to a scene
      std::string image_name;
      if(params().has_path("image_name"))
      {
        image_name =  params()["image_name"].as_string();
      }
      else
      {
        image_name =  params()["image_prefix"].as_string();
        image_name = expand_family_name(image_name, cycle);
      }

      vtkh::Render render = vtkh::MakeRender(1024,
                                             1024,
                                             *bounds,
                                             v_domain_ids,
                                             image_name);

      renders->push_back(render);
    }
    set_output<std::vector<vtkh::Render>>(renders);
}

//-----------------------------------------------------------------------------
VTKHClip::VTKHClip()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHClip::~VTKHClip()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHClip::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_clip";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHClip::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = true;

    bool type_present = false;

    if(params.has_child("sphere"))
    {
      type_present = true;
    }
    else if(params.has_child("box"))
    {
      type_present = true;
    }
    else if(params.has_child("plane"))
    {
      type_present = true;
    }

    if(!type_present)
    {
        info["errors"].append() = "Missing required parameter. Clip must specify a 'sphere', 'box', or 'plane'";
        res = false;
    }
    else
    {

      if(params.has_child("sphere"))
      {
         res = check_numeric("sphere/center/x",params, info, true) && res;
         res = check_numeric("sphere/center/y",params, info, true) && res;
         res = check_numeric("sphere/center/z",params, info, true) && res;
         res = check_numeric("sphere/radius",params, info, true) && res;

      }
      else if(params.has_child("box"))
      {
         res = check_numeric("box/min/x",params, info, true) && res;
         res = check_numeric("box/min/y",params, info, true) && res;
         res = check_numeric("box/min/z",params, info, true) && res;
         res = check_numeric("box/max/x",params, info, true) && res;
         res = check_numeric("box/max/y",params, info, true) && res;
         res = check_numeric("box/max/z",params, info, true) && res;
      }
      else if(params.has_child("plane"))
      {
         res = check_numeric("plane/point/x",params, info, true) && res;
         res = check_numeric("plane/point/y",params, info, true) && res;
         res = check_numeric("plane/point/z",params, info, true) && res;
         res = check_numeric("plane/normal/x",params, info, true) && res;
         res = check_numeric("plane/normal/y",params, info, true) && res;
         res = check_numeric("plane/normal/z",params, info, true) && res;
      }
    }

    res = check_string("invert",params, info, false) && res;
    res = check_string("topology",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("invert");
    valid_paths.push_back("sphere/center/x");
    valid_paths.push_back("sphere/center/y");
    valid_paths.push_back("sphere/center/z");
    valid_paths.push_back("sphere/radius");
    valid_paths.push_back("box/min/x");
    valid_paths.push_back("box/min/y");
    valid_paths.push_back("box/min/z");
    valid_paths.push_back("box/max/x");
    valid_paths.push_back("box/max/y");
    valid_paths.push_back("box/max/z");
    valid_paths.push_back("plane/point/x");
    valid_paths.push_back("plane/point/y");
    valid_paths.push_back("plane/point/z");
    valid_paths.push_back("plane/normal/x");
    valid_paths.push_back("plane/normal/y");
    valid_paths.push_back("plane/normal/z");
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
VTKHClip::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHClip input must be a vtk-h dataset");
    }


    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Clip clipper;

    clipper.SetInput(data);

    if(params().has_path("sphere"))
    {
      const Node &sphere = params()["sphere"];
      double center[3];

      center[0] = sphere["center/x"].to_float64();
      center[1] = sphere["center/y"].to_float64();
      center[2] = sphere["center/z"].to_float64();
      double radius = sphere["radius"].to_float64();
      clipper.SetSphereClip(center, radius);
    }
    else if(params().has_path("box"))
    {
      const Node &box = params()["box"];
      vtkm::Bounds bounds;
      bounds.X.Min= box["min/x"].to_float64();
      bounds.Y.Min= box["min/y"].to_float64();
      bounds.Z.Min= box["min/z"].to_float64();
      bounds.X.Max = box["max/x"].to_float64();
      bounds.Y.Max = box["max/y"].to_float64();
      bounds.Z.Max = box["max/z"].to_float64();
      clipper.SetBoxClip(bounds);
    }
    else if(params().has_path("plane"))
    {
      const Node &plane= params()["plane"];
      double point[3], normal[3];;

      point[0] = plane["point/x"].to_float64();
      point[1] = plane["point/y"].to_float64();
      point[2] = plane["point/z"].to_float64();
      normal[0] = plane["normal/x"].to_float64();
      normal[1] = plane["normal/y"].to_float64();
      normal[2] = plane["normal/z"].to_float64();
      clipper.SetPlaneClip(point, normal);
    }

    if(params().has_child("invert"))
    {
      std::string invert = params()["invert"].as_string();
      if(invert == "true")
      {
        clipper.SetInvertClip(true);
      }
    }

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    set_output<vtkh::DataSet>(clip_output);
}

//-----------------------------------------------------------------------------
VTKHClipWithField::VTKHClipWithField()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHClipWithField::~VTKHClipWithField()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHClipWithField::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_clip_with_field";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHClipWithField::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_numeric("clip_value",params, info, true);
    res = check_string("field",params, info, true) && res;
    res = check_string("invert",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("clip_value");
    valid_paths.push_back("invert");
    valid_paths.push_back("field");
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
VTKHClipWithField::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHClipWithField input must be a vtk-h dataset");
    }


    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::ClipField clipper;

    clipper.SetInput(data);

    if(params().has_child("invert"))
    {
      std::string invert = params()["invert"].as_string();
      if(invert == "true")
      {
        clipper.SetInvertClip(true);
      }
    }

    vtkm::Float64 clip_value = params()["clip_value"].to_float64();
    std::string field_name = params()["field"].as_string();

    clipper.SetField(field_name);
    clipper.SetClipValue(clip_value);

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    set_output<vtkh::DataSet>(clip_output);
}

//-----------------------------------------------------------------------------
VTKHIsoVolume::VTKHIsoVolume()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHIsoVolume::~VTKHIsoVolume()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHIsoVolume::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_iso_volume";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHIsoVolume::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_numeric("min_value",params, info, true);
    res = check_numeric("max_value",params, info, true) && res;
    res = check_string("field",params, info, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    valid_paths.push_back("field");
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
VTKHIsoVolume::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHIsoVolume input must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::IsoVolume clipper;

    clipper.SetInput(data);

    vtkm::Range clip_range;
    clip_range.Min = params()["min_value"].to_float64();
    clip_range.Max = params()["max_value"].to_float64();
    std::string field_name = params()["field"].as_string();

    clipper.SetField(field_name);
    clipper.SetRange(clip_range);

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    set_output<vtkh::DataSet>(clip_output);
}


//-----------------------------------------------------------------------------
EnsureVTKM::EnsureVTKM()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
EnsureVTKM::~EnsureVTKM()
{
// empty
}

//-----------------------------------------------------------------------------
void
EnsureVTKM::declare_interface(Node &i)
{
    i["type_name"]   = "ensure_vtkm";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
EnsureVTKM::execute()
{
#if !defined(ASCENT_VTKM_ENABLED)
        ASCENT_ERROR("ascent was not built with VTKm support!");
#else
    if(input(0).check_type<vtkm::cont::DataSet>())
    {
        set_output(input(0));
    }
    else if(input(0).check_type<Node>())
    {
        bool zero_copy = false;
        // convert from conduit to vtkm
        const Node *n_input = input<Node>(0);
        vtkm::cont::DataSet  *res = VTKHDataAdapter::BlueprintToVTKmDataSet(*n_input, zero_copy);
        set_output<vtkm::cont::DataSet>(res);
    }
    else
    {
        ASCENT_ERROR("unsupported input type for ensure_vtkm");
    }
#endif
}


//-----------------------------------------------------------------------------
VTKHBounds::VTKHBounds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHBounds::~VTKHBounds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHBounds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_bounds";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
VTKHBounds::execute()
{
    vtkm::Bounds *bounds = new vtkm::Bounds;

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("in must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    bounds->Include(data->GetGlobalBounds());
    set_output<vtkm::Bounds>(bounds);
}


//-----------------------------------------------------------------------------
VTKHUnionBounds::VTKHUnionBounds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHUnionBounds::~VTKHUnionBounds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHUnionBounds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_union_bounds";
    i["port_names"].append() = "a";
    i["port_names"].append() = "b";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
VTKHUnionBounds::execute()
{
    if(!input(0).check_type<vtkm::Bounds>())
    {
        ASCENT_ERROR("'a' must be a vtkm::Bounds * instance");
    }

    if(!input(1).check_type<vtkm::Bounds>())
    {
        ASCENT_ERROR("'b' must be a vtkm::Bounds * instance");
    }

    vtkm::Bounds *result = new vtkm::Bounds;

    vtkm::Bounds *bounds_a = input<vtkm::Bounds>(0);
    vtkm::Bounds *bounds_b = input<vtkm::Bounds>(1);

    result->Include(*bounds_a);
    result->Include(*bounds_b);
    set_output<vtkm::Bounds>(result);
}



//-----------------------------------------------------------------------------
VTKHDomainIds::VTKHDomainIds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHDomainIds::~VTKHDomainIds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHDomainIds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_domain_ids";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
VTKHDomainIds::execute()
{
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("'in' must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);

    std::vector<vtkm::Id> domain_ids = data->GetDomainIds();

    std::set<vtkm::Id> *result = new std::set<vtkm::Id>;
    result->insert(domain_ids.begin(), domain_ids.end());

    set_output<std::set<vtkm::Id> >(result);
}



//-----------------------------------------------------------------------------
VTKHUnionDomainIds::VTKHUnionDomainIds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHUnionDomainIds::~VTKHUnionDomainIds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHUnionDomainIds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_union_domain_ids";
    i["port_names"].append() = "a";
    i["port_names"].append() = "b";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void
VTKHUnionDomainIds::execute()
{
    if(!input(0).check_type<std::set<vtkm::Id> >())
    {
        ASCENT_ERROR("'a' must be a std::set<vtkm::Id> * instance");
    }

    if(!input(1).check_type<std::set<vtkm::Id> >())
    {
        ASCENT_ERROR("'b' must be a std::set<vtkm::Id> * instance");
    }


    std::set<vtkm::Id> *dids_a = input<std::set<vtkm::Id>>(0);
    std::set<vtkm::Id> *dids_b = input<std::set<vtkm::Id>>(1);

    std::set<vtkm::Id> *result = new std::set<vtkm::Id>;
    *result = *dids_a;

    result->insert(dids_b->begin(), dids_b->end());

    set_output<std::set<vtkm::Id>>(result);
}

//-----------------------------------------------------------------------------
AddPlot::AddPlot()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
AddPlot::~AddPlot()
{
// empty
}

//-----------------------------------------------------------------------------
void
AddPlot::declare_interface(Node &i)
{
    i["type_name"] = "add_plot";
    i["port_names"].append() = "scene";
    i["port_names"].append() = "plot";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void
AddPlot::execute()
{
    if(!input(0).check_type<detail::AscentScene>())
    {
        ASCENT_ERROR("'scene' must be a AscentScene * instance");
    }

    if(!input(1).check_type<detail::RendererContainer >())
    {
        ASCENT_ERROR("'plot' must be a detail::RendererContainer * instance");
    }

    detail::AscentScene *scene = input<detail::AscentScene>(0);
    detail::RendererContainer * cont = input<detail::RendererContainer>(1);
    scene->AddRenderer(cont);
    set_output<detail::AscentScene>(scene);
}

//-----------------------------------------------------------------------------
CreatePlot::CreatePlot()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
CreatePlot::~CreatePlot()
{
// empty
}

//-----------------------------------------------------------------------------
void
CreatePlot::declare_interface(Node &i)
{
    i["type_name"] = "create_plot";
    i["port_names"].append() = "a";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
bool
CreatePlot::verify_params(const conduit::Node &params,
                          conduit::Node &info)
{
    info.reset();

    bool res = check_string("type",params, info, true);

    bool is_mesh = false;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("type");
    valid_paths.push_back("pipeline");

    if(res)
    {
      if(params["type"].as_string() == "mesh")
      {
        is_mesh = true;
      }
    }

    if(!is_mesh)
    {
      res &= check_string("field", params, info, true);
      valid_paths.push_back("field");
      valid_paths.push_back("points/radius");
      valid_paths.push_back("points/radius_delta");
      valid_paths.push_back("min_value");
      valid_paths.push_back("max_value");
    }
    else
    {
      valid_paths.push_back("overlay");
      valid_paths.push_back("show_internal");
    }


    std::vector<std::string> ignore_paths;
    ignore_paths.push_back("color_table");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::check_color_table_surprises(params["color_table"]);
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
CreatePlot::execute()
{
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("create_plot input must be a vtk-h dataset");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);

    conduit::Node plot_params = params();
    std::string type = params()["type"].as_string();

    if(data->GlobalIsEmpty())
    {
      ASCENT_INFO(type<<" plot yielded no data, i.e., no cells remain");
    }

    vtkh::Renderer *renderer = nullptr;

    if(type == "pseudocolor")
    {
      bool is_point_mesh = data->IsPointMesh();
      if(is_point_mesh)
      {
        vtkh::PointRenderer *p_renderer = new vtkh::PointRenderer();
        p_renderer->UseCells();
        if(plot_params.has_path("points/radius"))
        {
          float radius = plot_params["points/radius"].to_float32();
          p_renderer->SetBaseRadius(radius);
        }
        // default is to use a constant radius
        // if the radius delta is present, we will
        // vary radii based on the scalar value
        if(plot_params.has_path("points/radius_delta"))
        {
          float radius = plot_params["points/radius_delta"].to_float32();
          p_renderer->UseVariableRadius(true);
          p_renderer->SetRadiusDelta(radius);
        }
        renderer = p_renderer;
      }
      else
      {
        renderer = new vtkh::RayTracer();
      }

    }
    else if(type == "volume")
    {
      renderer = new vtkh::VolumeRenderer();
    }
    else if(type == "mesh")
    {
      renderer = new vtkh::MeshRenderer();
    }
    else
    {
        ASCENT_ERROR("create_plot unknown plot type '"<<type<<"'");
    }

    // get the plot params
    if(plot_params.has_path("color_table"))
    {
      vtkm::cont::ColorTable color_table =  parse_color_table(plot_params["color_table"]);
      renderer->SetColorTable(color_table);
    }

    vtkm::Range scalar_range;
    if(plot_params.has_path("min_value"))
    {
      scalar_range.Min = plot_params["min_value"].to_float64();
    }

    if(plot_params.has_path("max_value"))
    {
      scalar_range.Max = plot_params["max_value"].to_float64();
    }

    renderer->SetRange(scalar_range);

    if(plot_params.has_path("field"))
    {
      std::string field_name = plot_params["field"].as_string();
      if(!data->GlobalFieldExists(field_name))
      {
        ASCENT_INFO("Plot variable '"<<field_name<<"' does not exist");
      }
      renderer->SetField(field_name);
    }

    if(type == "mesh")
    {
      vtkh::MeshRenderer *mesh = dynamic_cast<vtkh::MeshRenderer*>(renderer);
      if(!plot_params.has_path("field"))
      {
        // The renderer needs a field, so add one if
        // needed. This will eventually go away once
        // the mesh mapper in vtkm can handle no field
        const std::string fname = "constant_mesh_field";
        data->AddConstantPointField(0.f, fname);
        renderer->SetField(fname);
        mesh->SetUseForegroundColor(true);
      }

      mesh->SetIsOverlay(true);
      if(plot_params.has_path("overlay"))
      {
        if(plot_params["overlay"].as_string() == "false")
        {
          mesh->SetIsOverlay(false);
        }
      }

      if(plot_params.has_path("show_internal"))
      {
        if(plot_params["show_internal"].as_string() == "true")
        {
          mesh->SetShowInternal(true);
        }
      }
    } // is mesh

    std::string key = this->name() + "_cont";

    renderer->SetInput(data);

    detail::RendererContainer *container = new detail::RendererContainer(key,
                                                                         &graph().workspace().registry(),
                                                                         renderer);
    set_output<detail::RendererContainer>(container);

}


//-----------------------------------------------------------------------------
CreateScene::CreateScene()
: Filter()
{}

//-----------------------------------------------------------------------------
CreateScene::~CreateScene()
{}

//-----------------------------------------------------------------------------
void
CreateScene::declare_interface(Node &i)
{
    i["type_name"]   = "create_scene";
    i["output_port"] = "true";
    i["port_names"] = DataType::empty();
}

//-----------------------------------------------------------------------------
void
CreateScene::execute()
{
    detail::AscentScene *scene = new detail::AscentScene(&graph().workspace().registry());
    set_output<detail::AscentScene>(scene);
}

//-----------------------------------------------------------------------------
ExecScene::ExecScene()
  : Filter()
{

}

//-----------------------------------------------------------------------------
ExecScene::~ExecScene()
{

}

//-----------------------------------------------------------------------------
void
ExecScene::declare_interface(conduit::Node &i)
{
    i["type_name"] = "exec_scene";
    i["port_names"].append() = "scene";
    i["port_names"].append() = "renders";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
void
ExecScene::execute()
{
    if(!input(0).check_type<detail::AscentScene>())
    {
        ASCENT_ERROR("'scene' must be a AscentScene * instance");
    }

    if(!input(1).check_type<std::vector<vtkh::Render> >())
    {
        ASCENT_ERROR("'renders' must be a std::vector<vtkh::Render> * instance");
    }

    detail::AscentScene *scene = input<detail::AscentScene>(0);
    std::vector<vtkh::Render> * renders = input<std::vector<vtkh::Render>>(1);
    scene->Execute(*renders);

    // the images should exist now so add them to the image list
    // this can be used for the web server or jupyter

    if(!graph().workspace().registry().has_entry("image_list"))
    {
      conduit::Node *image_list = new conduit::Node();
      graph().workspace().registry().add<Node>("image_list", image_list,1);
    }

    conduit::Node *image_list = graph().workspace().registry().fetch<Node>("image_list");
    for(int i = 0; i < renders->size(); ++i)
    {
      const std::string image_name = renders->at(i).GetImageName() + ".png";
      conduit::Node image_data;
      image_data["image_name"] = image_name;
      image_data["image_width"] = renders->at(i).GetWidth();
      image_data["image_height"] = renders->at(i).GetHeight();

      image_data["camera/position"].set(&renders->at(i).GetCamera().GetPosition()[0],3);
      image_data["camera/look_at"].set(&renders->at(i).GetCamera().GetLookAt()[0],3);
      image_data["camera/up"].set(&renders->at(i).GetCamera().GetViewUp()[0],3);
      image_data["camera/zoom"] = renders->at(i).GetCamera().GetZoom();
      image_data["camera/fov"] = renders->at(i).GetCamera().GetFieldOfView();
      vtkm::Bounds bounds=  renders->at(i).GetSceneBounds();
      double coord_bounds [6] = {bounds.X.Min,
                                 bounds.Y.Min,
                                 bounds.Z.Min,
                                 bounds.X.Max,
                                 bounds.Y.Max,
                                 bounds.Z.Max};

      image_data["scene_bounds"].set(coord_bounds, 6);

      image_list->append() = image_data;
    }

}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

VTKHLagrangian::VTKHLagrangian()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLagrangian::~VTKHLagrangian()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLagrangian::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_lagrangian";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHLagrangian::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("step_size", params, info, true);
    res &= check_numeric("write_frequency", params, info, true);
    res &= check_numeric("cust_res", params, info, true);
    res &= check_numeric("x_res", params, info, true);
    res &= check_numeric("y_res", params, info, true);
    res &= check_numeric("z_res", params, info, true);


    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("step_size");
    valid_paths.push_back("write_frequency");
    valid_paths.push_back("cust_res");
    valid_paths.push_back("x_res");
    valid_paths.push_back("y_res");
    valid_paths.push_back("z_res");

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
VTKHLagrangian::execute()
{
    vtkh::DataSet *data = nullptr;
    if(input(0).check_type<vtkh::DataSet>())
    {
      data = input<vtkh::DataSet>(0);
    }
    else if(input(0).check_type<Node>())
    {
      const Node *n_input = input<Node>(0);
      data = VTKHDataAdapter::BlueprintToVTKHDataSet(*n_input);
    }
    else
    {
        ASCENT_ERROR("vtkh_lagrangian input must be a< vtkh::DataSet> or <Node>");
    }

    std::string field_name = params()["field"].as_string();
    double step_size = params()["step_size"].to_float64();
    int write_frequency = params()["write_frequency"].to_int32();
    int cust_res = params()["cust_res"].to_int32();
    int x_res = params()["x_res"].to_int32();
    int y_res = params()["y_res"].to_int32();
    int z_res = params()["z_res"].to_int32();

    vtkh::Lagrangian lagrangian;

    lagrangian.SetInput(data);
    lagrangian.SetField(field_name);
    lagrangian.SetStepSize(step_size);
    lagrangian.SetWriteFrequency(write_frequency);
    lagrangian.SetCustomSeedResolution(cust_res);
    lagrangian.SetSeedResolutionInX(x_res);
    lagrangian.SetSeedResolutionInY(y_res);
    lagrangian.SetSeedResolutionInZ(z_res);
    lagrangian.Update();

    vtkh::DataSet *lagrangian_output = lagrangian.GetOutput();

    set_output<vtkh::DataSet>(lagrangian_output);
}

//-----------------------------------------------------------------------------

VTKHLog::VTKHLog()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLog::~VTKHLog()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLog::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_log";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHLog::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_numeric("clamp_min_value",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("clamp_min_value");

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
VTKHLog::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_log input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Log logger;
    logger.SetInput(data);
    logger.SetField(field_name);
    if(params().has_path("output_name"))
    {
      logger.SetResultField(params()["output_name"].as_string());
    }

    if(params().has_path("clamp_min_value"))
    {
      logger.SetClampMin(params()["clamp_min_value"].to_float32());
      logger.SetClampToMin(true);
    }

    logger.Update();

    vtkh::DataSet *log_output = logger.GetOutput();

    set_output<vtkh::DataSet>(log_output);
}

//-----------------------------------------------------------------------------

VTKHRecenter::VTKHRecenter()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHRecenter::~VTKHRecenter()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHRecenter::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_recenter";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHRecenter::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("association",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("association");

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
VTKHRecenter::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
      ASCENT_ERROR("vtkh_recenter input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    std::string association = params()["association"].as_string();
    if(association != "vertex" && association != "element")
    {
      ASCENT_ERROR("Recenter: resulting field association '"<<association<<"'"
                   <<" must have a value of 'vertex' or 'element'");
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Recenter recenter;

    recenter.SetInput(data);
    recenter.SetField(field_name);

    if(association == "vertex")
    {
      recenter.SetResultAssoc(vtkm::cont::Field::Association::POINTS);
    }
    if(association == "element")
    {
      recenter.SetResultAssoc(vtkm::cont::Field::Association::CELL_SET);
    }

    recenter.Update();

    vtkh::DataSet *recenter_output = recenter.GetOutput();

    set_output<vtkh::DataSet>(recenter_output);
}
//-----------------------------------------------------------------------------

VTKHHistSampling::VTKHHistSampling()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHHistSampling::~VTKHHistSampling()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHHistSampling::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_hist_sampling";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHHistSampling::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("bins",params, info, false);
    res &= check_numeric("sample_rate",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("bins");
    valid_paths.push_back("sample_rate");

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
VTKHHistSampling::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_hist_sampling input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    float sample_rate = .1f;
    if(params().has_path("sample_rate"))
    {
      sample_rate = params()["sample_rate"].to_float32();
      if(sample_rate <= 0.f || sample_rate >= 1.f)
      {
        ASCENT_ERROR("vtkh_hist_sampling 'sample_rate' value '"<<sample_rate<<"'"
                     <<" not in the range (0,1)");
      }
    }

    int bins = 128;

    if(params().has_path("bins"))
    {
      bins = params()["bins"].to_int32();
      if(bins <= 0.f)
      {
        ASCENT_ERROR("vtkh_hist_sampling 'bins' value '"<<bins<<"'"
                     <<" must be positive");
      }
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);

    // TODO: write helper functions for this
    std::string ghost_field = "";
    Node * meta = graph().workspace().registry().fetch<Node>("metadata");
    if(meta->has_path("ghost_field"))
    {
      ghost_field = (*meta)["ghost_field"].as_string();
      if(!data->GlobalFieldExists(ghost_field))
      {
        // can't find it
        ghost_field = "";
      }

    }

    vtkh::HistSampling hist;

    hist.SetInput(data);
    hist.SetField(field_name);
    hist.SetNumBins(bins);
    hist.SetSamplingPercent(sample_rate);
    if(ghost_field != "")
    {
      hist.SetGhostField(ghost_field);
    }

    hist.Update();
    vtkh::DataSet *hist_output = hist.GetOutput();

    set_output<vtkh::DataSet>(hist_output);
}

//-----------------------------------------------------------------------------

VTKHQCriterion::VTKHQCriterion()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHQCriterion::~VTKHQCriterion()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHQCriterion::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_qcriterion";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHQCriterion::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
VTKHQCriterion::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_qcriterion input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Gradient grad;
    grad.SetInput(data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_qcriterion = true;

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }
    if(params().has_path("output_name"))
    {
      grad_params.qcriterion_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    set_output<vtkh::DataSet>(grad_output);
}
//-----------------------------------------------------------------------------

VTKHDivergence::VTKHDivergence()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHDivergence::~VTKHDivergence()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHDivergence::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_divergence";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHDivergence::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
VTKHDivergence::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_divergence input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Gradient grad;
    grad.SetInput(data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_divergence = true;

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.divergence_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    set_output<vtkh::DataSet>(grad_output);
}
//-----------------------------------------------------------------------------

VTKHVorticity::VTKHVorticity()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVorticity::~VTKHVorticity()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVorticity::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_curl";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHVorticity::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
VTKHVorticity::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_vorticity input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Gradient grad;
    grad.SetInput(data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_vorticity = true;

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.vorticity_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    set_output<vtkh::DataSet>(grad_output);
}
//-----------------------------------------------------------------------------

VTKHGradient::VTKHGradient()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHGradient::~VTKHGradient()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHGradient::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_gradient";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHGradient::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
VTKHGradient::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_gradient input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Gradient grad;
    grad.SetInput(data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.output_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    set_output<vtkh::DataSet>(grad_output);
}

//-----------------------------------------------------------------------------

VTKHStats::VTKHStats()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHStats::~VTKHStats()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHStats::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_stats";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
VTKHStats::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");

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
VTKHStats::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_stats input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Statistics stats;

    vtkh::Statistics::Result res = stats.Run(*data, field_name);
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank == 0)
    {
      res.Print(std::cout);
    }
}
//-----------------------------------------------------------------------------

VTKHHistogram::VTKHHistogram()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHHistogram::~VTKHHistogram()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHHistogram::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_histogram";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
VTKHHistogram::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("bins",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("bins");

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
VTKHHistogram::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_histogram input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    int bins = 128;
    if(params().has_path("bins"))
    {
      bins = params()["bins"].to_int32();
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Histogram hist;

    hist.SetNumBins(bins);
    vtkh::Histogram::HistogramResult res = hist.Run(*data, field_name);
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank == 0)
    {
      res.Print(std::cout);
    }
}
//-----------------------------------------------------------------------------

VTKHParticleAdvection::VTKHParticleAdvection()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHParticleAdvection::~VTKHParticleAdvection()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHParticleAdvection::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_particle_advection";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHParticleAdvection::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("seeds",params, info, false);
    res &= check_numeric("step_size",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("seeds");
    valid_paths.push_back("step_size");

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
VTKHParticleAdvection::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_particle_advection input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    float step_size = 0.1f;
    int seeds = 500;
    if(params().has_path("seeds"))
    {
      seeds = params()["seeds"].to_int32();
    }
    if(params().has_path("step_size"))
    {
      step_size = params()["step_size"].to_float32();
    }

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::ParticleAdvection streamline;

    streamline.SetInput(data);
    streamline.SetField(field_name);
    streamline.SetStepSize(step_size);
    streamline.SetSeedsRandomWhole(seeds);

    streamline.Update();

    vtkh::DataSet *output = streamline.GetOutput();
    set_output<vtkh::DataSet>(output);
}

//-----------------------------------------------------------------------------

VTKHNoOp::VTKHNoOp()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHNoOp::~VTKHNoOp()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHNoOp::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_no_op";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHNoOp::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");

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
VTKHNoOp::execute()
{

    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_no_op input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::NoOp noop;

    noop.SetInput(data);
    noop.SetField(field_name);

    noop.Update();

    vtkh::DataSet *noop_output = noop.GetOutput();
    set_output<vtkh::DataSet>(noop_output);
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
