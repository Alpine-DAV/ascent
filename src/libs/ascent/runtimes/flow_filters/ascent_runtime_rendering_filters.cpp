//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_rendering_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_rendering_filters.hpp"

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
#include <ascent_metadata.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_data_object.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_web_interface.hpp> // -- for web_client_root_directory()
#include <ascent_resources.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <ascent_vtkh_collection.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/rendering/AutoCamera.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_vtkh_utils.hpp>
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
  valid_paths.push_back("annotation");
  valid_paths.push_back("discrete");

  std::vector<std::string> ignore_paths;
  ignore_paths.push_back("control_points");

  surprises += surprise_check(valid_paths, ignore_paths, color_table);
  if(color_table.has_path("control_points"))
  {
    const Node &control_points_node = color_table.fetch("control_points");

    if (control_points_node.dtype().is_list())
    {
        // Valid path options for the expanded control points input format
        std::vector<std::string> c_valid_paths;
        c_valid_paths.push_back("type");
        c_valid_paths.push_back("alpha");
        c_valid_paths.push_back("color");
        c_valid_paths.push_back("position");

        const int num_points = control_points_node.number_of_children();
        for(int i = 0; i < num_points; ++i)
        {
            const conduit::Node &point = control_points_node.child(i);
            surprises += surprise_check(c_valid_paths, point);
        }
    }
    else if (control_points_node.dtype().is_object())
    {
        // Valid path options for the compressed control points input format
        std::vector<std::string> c_valid_paths;
        c_valid_paths.push_back("r");
        c_valid_paths.push_back("g");
        c_valid_paths.push_back("b");
        c_valid_paths.push_back("a");
        c_valid_paths.push_back("position");


        surprises += surprise_check(c_valid_paths, control_points_node);
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
  r_valid_paths.push_back("camera/2d");
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
  r_valid_paths.push_back("phi_range");
  r_valid_paths.push_back("dphi");
  r_valid_paths.push_back("phi_num_angles");
  r_valid_paths.push_back("phi_angles");
  r_valid_paths.push_back("theta");
  r_valid_paths.push_back("theta_range");
  r_valid_paths.push_back("dtheta");
  r_valid_paths.push_back("theta_num_angles");
  r_valid_paths.push_back("theta_angles");
  r_valid_paths.push_back("phi_theta_positions");
  r_valid_paths.push_back("db_name");
  r_valid_paths.push_back("output_path");
  r_valid_paths.push_back("render_bg");
  r_valid_paths.push_back("annotations");
  r_valid_paths.push_back("world_annotations");
  r_valid_paths.push_back("screen_annotations");
  r_valid_paths.push_back("axis_scale_x");
  r_valid_paths.push_back("axis_scale_y");
  r_valid_paths.push_back("axis_scale_z");
  r_valid_paths.push_back("fg_color");
  r_valid_paths.push_back("bg_color");
  r_valid_paths.push_back("shading");
  r_valid_paths.push_back("use_original_bounds");
  r_valid_paths.push_back("dataset_bounds");
  r_valid_paths.push_back("auto_camera/metric");
  r_valid_paths.push_back("auto_camera/field");
  r_valid_paths.push_back("auto_camera/samples");
  r_valid_paths.push_back("auto_camera/bins");
  r_valid_paths.push_back("auto_camera/height");
  r_valid_paths.push_back("auto_camera/width");
  r_valid_paths.push_back("color_bar_position");

  std::vector<std::string> r_ignore_paths;
  r_ignore_paths.push_back("phi_theta_positions");

  for(int i = 0; i < num_renders; ++i)
  {
    const conduit::Node &render_node = renders_node.child(i);
    surprises += surprise_check(r_valid_paths, r_ignore_paths, render_node);

    if(render_node.has_path("phi_theta_positions"))
    {
      const conduit::Node &phi_theta_positions = render_node["phi_theta_positions"];
      const int num_positions = phi_theta_positions.number_of_children();
      for(int i = 0; i < num_positions; ++i)
      {
        const conduit::Node &position = phi_theta_positions.child(i);
        std::stringstream ss;
        ss << "[" << i << "]";
        if (position.name() != ss.str())
        {
          surprises += "Surprise parameter '";
          surprises += position.name();
          surprises += "'\n";
        }
      }
    }
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
  // make sure the data set we need does not get deleted
  // out from under us, which will happen
  std::shared_ptr<VTKHCollection> m_collection;
  std::string m_topo_name;
  bool m_valid;
  // we have to keep the data object that spit out
  // the vtkh collection we are rendering since
  // this could be a temporary result created by a
  // pipeline. If we dont' keep it, then it will
  // be freed before we can render it
  DataObject m_data;
public:
  RendererContainer()
   : m_valid(false)
  {};
  RendererContainer(std::string key,
                    flow::Registry *r,
                    vtkh::Renderer *renderer,
                    std::shared_ptr<VTKHCollection> collection,
                    std::string topo_name,
                    DataObject &data_object)
    : m_key(key),
      m_registry(r),
      m_collection(collection),
      m_topo_name(topo_name),
      m_valid(true),
      m_data(data_object)
  {
    // we have to keep around the dataset so we bring the
    // whole collection with us
    vtkh::DataSet &data = m_collection->dataset_by_topology(m_topo_name);
    renderer->SetInput(&data);
    m_registry->add<vtkh::Renderer>(m_key,renderer,1);
  }

  bool is_valid()
  {
    return m_valid;
  }

  vtkh::Renderer *
  Fetch()
  {
    return m_registry->fetch<vtkh::Renderer>(m_key);
  }

  ~RendererContainer()
  {
    // we reset the registry in the runtime
    // which will automatically delete this pointer
    // m_registry->consume(m_key);
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
                                         image_name);
  Node meta = Metadata::n_metadata;
  if(meta.has_path("comments"))
  {
    const conduit::Node comments_node = meta["comments"];
    const int num_comments = comments_node.number_of_children();
    std::vector<std::string> comments;
    for(int i = 0; i < num_comments; ++i)
    {
      comments.push_back(comments_node.child(i).to_yaml());
    }
    render.SetComments(comments);
  }

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
  if(render_node.has_path("shading"))
  {
    bool on = render_node["shading"].as_string() == "enabled";
    render.SetShadingOn(on);
  }

  bool annot_all_off = false;
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
      annot_all_off = true;
    }
  }

  if(!annot_all_off && render_node.has_path("world_annotations"))
  {
    if(!render_node["world_annotations"].dtype().is_string())
    {
      ASCENT_ERROR("render/world_annotations node must be a string value");
    }
    const std::string annot = render_node["world_annotations"].as_string();
    // default is always render world annotations
    if(annot == "false")
    {
      render.DoRenderWorldAnnotations(false);
    }
  }

  if(!annot_all_off && render_node.has_path("screen_annotations"))
  {
    if(!render_node["screen_annotations"].dtype().is_string())
    {
      ASCENT_ERROR("render/screen_annotations node must be a string value");
    }
    const std::string annot = render_node["screen_annotations"].as_string();
    // default is always render screen annotations
    if(annot == "false")
    {
      render.DoRenderScreenAnnotations(false);
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

  float axis_scale_x = 1.f;
  float axis_scale_y = 1.f;
  float axis_scale_z = 1.f;

  if(render_node.has_path("axis_scale_x"))
  {
    axis_scale_x = render_node["axis_scale_x"].to_float32();
  }

  if(render_node.has_path("axis_scale_y"))
  {
    axis_scale_y = render_node["axis_scale_y"].to_float32();
  }

  if(render_node.has_path("axis_scale_z"))
  {
    axis_scale_z = render_node["axis_scale_z"].to_float32();
  }

  render.ScaleWorldAnnotations(axis_scale_x, axis_scale_y, axis_scale_z);

  if(render_node.has_path("color_bar_position"))
  {
    if(!render_node["color_bar_position"].dtype().is_number() ||
		    render_node["color_bar_position"].dtype().number_of_elements()%4 != 0)
    {
      ASCENT_ERROR("render/color_bar_position must be an array of 4 values for each color bar");
    }

    int positions = render_node["color_bar_position"].dtype().number_of_elements()/4;
    std::vector<vtkm::Bounds> cb_position;
    for(int i = 0; i < positions; i++)
    {
      conduit::Node n;
      render_node["color_bar_position"].to_float32_array(n);
      const float32 *cb_pos = n.as_float32_ptr();
      vtkm::Bounds pos(vtkm::Range(cb_pos[0+4*i],cb_pos[1+4*i]),
			vtkm::Range(cb_pos[2+4*i],cb_pos[3+4*i]),
			vtkm::Range(0.0,0.0));
      cb_position.push_back(pos);
    }
    render.SetColorBarPosition(cb_position);

  }

  return render;
}

class CinemaManager
{
protected:
  std::vector<vtkm::rendering::Camera> m_cameras;
  std::vector<std::string>             m_image_names;
  std::vector<std::tuple<float,float>> m_camera_angles;
  std::vector<float>                   m_phi_values;
  std::vector<float>                   m_theta_values;
  std::vector<float>                   m_times;
  std::string                          m_csv;

  vtkm::Bounds                         m_bounds;
  int                                  m_phi;
  float                                m_phi_min;
  float                                m_phi_inc;
  int                                  m_theta;
  float                                m_theta_min;
  float                                m_theta_inc;
  std::string                          m_image_name;
  std::string                          m_image_path;
  std::string                          m_db_path;
  std::string                          m_base_path;
  float                                m_time;
public:
  CinemaManager(vtkm::Bounds bounds,
                const conduit::Node &render_node,
                const std::string image_name,
                const std::string path)
    : m_bounds(bounds),
      m_image_name(image_name),
      m_time(0.f)
  {
    if(render_node.has_path("phi_theta_positions"))
    {
      const conduit::Node &positions = render_node["phi_theta_positions"];
      for (int p = 0; p < positions.number_of_children(); ++p)
      {
        float64_accessor phi_theta = positions[p].as_float64_accessor();
        if (phi_theta.number_of_elements() != 2)
        {
          ASCENT_ERROR("Cinema camera phi_theta_positions must be an array of tuples");
        }
        std::tuple<float,float> angles(phi_theta[0], phi_theta[1]);
        m_camera_angles.push_back(angles);
      }
    }
    else
    {
      // Handle phi.
      if(render_node.has_path("phi"))
      {
        m_phi_min = -180.0;
        m_phi = render_node["phi"].to_int32();
        m_phi_inc = 360.0 / double(m_phi);
        this->create_angles_from_range(m_phi_values, m_phi_min, m_phi_inc, m_phi);
      }
      else if(render_node.has_path("phi_angles"))
      {
        float64_accessor phi_angles = render_node["phi_angles"].as_float64_accessor();
        for(int p = 0; p < phi_angles.number_of_elements(); ++p)
        {
          m_phi_values.push_back(phi_angles[p]);
        }
      }
      else if(render_node.has_path("phi_range") && render_node.has_path("dphi"))
      {
        float64_accessor phi_range = render_node["phi_range"].as_float64_accessor();
        if (phi_range[0] >= phi_range[1])
        {
          ASCENT_ERROR("Cinema camera phi_range[0] must be less that phi_range[1]");
        }
        m_phi_min = phi_range[0];
        m_phi_inc = render_node["dphi"].to_float64();
        m_phi =  int(floor((phi_range[1] - phi_range[0] + 1.0e-6) / m_phi_inc)) + 1;
        this->create_angles_from_range(m_phi_values, m_phi_min, m_phi_inc, m_phi);
      }
      else if(render_node.has_path("phi_range") && render_node.has_path("phi_num_angles"))
      {
        float64_accessor phi_range = render_node["phi_range"].as_float64_accessor();
        if (phi_range[0] >= phi_range[1])
        {
          ASCENT_ERROR("Cinema camera phi_range[0] must be less that phi_range[1]");
        }
        m_phi_min = phi_range[0];
        m_phi = render_node["phi_num_angles"].to_int32();
        m_phi_inc =  (phi_range[1] - phi_range[0]) / double(m_phi - 1);
        this->create_angles_from_range(m_phi_values, m_phi_min, m_phi_inc, m_phi);
      }
      else
      {
        ASCENT_ERROR("Cinema camera must specify phi or phi_range");
      }

      // Handle theta.
      if(render_node.has_path("theta"))
      {
        m_theta_min = 0.0;
        m_theta = render_node["theta"].to_int32();
        m_theta_inc = 180.0 / double(m_theta);
        this->create_angles_from_range(m_theta_values, m_theta_min, m_theta_inc, m_theta);
      }
      else if(render_node.has_path("theta_angles"))
      {
        float64_accessor theta_angles = render_node["theta_angles"].as_float64_accessor();
        for(int t = 0; t < theta_angles.number_of_elements(); ++t)
        {
          m_theta_values.push_back(theta_angles[t]);
        }
      }
      else if(render_node.has_path("theta_range") && render_node.has_path("dtheta"))
      {
        float64_accessor theta_range = render_node["theta_range"].as_float64_accessor();
        if (theta_range[0] >= theta_range[1])
        {
          ASCENT_ERROR("Cinema camera theta_range[0] must be less that theta_range[1]");
        }
        m_theta_min = theta_range[0];
        m_theta_inc = render_node["dtheta"].to_float64();
        m_theta =  int(floor((theta_range[1] - theta_range[0] + 1.0e-6) / m_theta_inc)) + 1;
        this->create_angles_from_range(m_theta_values, m_theta_min, m_theta_inc, m_theta);
      }
      else if(render_node.has_path("theta_range") && render_node.has_path("theta_num_angles"))
      {
        float64_accessor theta_range = render_node["theta_range"].as_float64_accessor();
        if (theta_range[0] >= theta_range[1])
        {
          ASCENT_ERROR("Cinema camera theta_range[0] must be less that theta_range[1]");
        }
        m_theta_min = theta_range[0];
        m_theta = render_node["theta_num_angles"].to_int32();
        m_theta_inc =  (theta_range[1] - theta_range[0]) / double(m_theta - 1);
        this->create_angles_from_range(m_theta_values, m_theta_min, m_theta_inc, m_theta);
      }
      else
      {
        ASCENT_ERROR("Cinema camera must specify theta or theta_range");
      }

      this->create_cinema_angles();
    }

    this->create_cinema_cameras(bounds);
    m_csv = "phi,theta,time,FILE\n";

    m_base_path = conduit::utils::join_file_path(path, "cinema_databases");
  }

  CinemaManager()
  {
    ASCENT_ERROR("Cannot create un-initialized CinemaManger");
  }

  std::string db_path()
  {
       return conduit::utils::join_file_path(m_base_path, m_image_name);
  }

  void set_bounds(vtkm::Bounds &bounds)
  {
    if(bounds != m_bounds)
    {
      this->create_cinema_cameras(bounds);
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

  void fill_renders(std::vector<vtkh::Render> *renders,
                    const conduit::Node &render_node)
  {
    conduit::Node render_copy = render_node;

    // allow zoom to be adjusted
    conduit::Node zoom;
    if(render_copy.has_path("camera/zoom"))
    {
        float zoom_val = render_node["camera/zoom"].to_value();
        if (zoom_val <= 0) {
            ASCENT_ERROR("Zoom must be greater than zero. Default zoom is 1.\n");
        }

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

    const int phi_size = m_phi_values.size();
    if (phi_size > 0)
    {
      conduit::Node phis;
      phis["default"] = get_string(m_phi_values[0]);
      phis["label"] = "phi";
      phis["type"] = "range";
      for(int i = 0; i < phi_size; ++i)
      {
        phis["values"].append().set(get_string(m_phi_values[i]));
      }

      meta["arguments/phi"] = phis;
    }

    const int theta_size = m_theta_values.size();
    if (theta_size > 0)
    {
      conduit::Node thetas;
      thetas["default"] = get_string(m_theta_values[0]);
      thetas["label"] = "theta";
      thetas["type"] = "range";
      for(int i = 0; i < theta_size; ++i)
      {
        thetas["values"].append().set(get_string(m_theta_values[i]));
      }

      meta["arguments/theta"] = thetas;
    }

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
    for(int a = 0; a < m_camera_angles.size(); ++a)
    {
      std::string phi = get_string(std::get<0>(m_camera_angles[a]));
      std::string theta = get_string(std::get<1>(m_camera_angles[a]));
      csv<<phi<<",";
      csv<<theta<<",";
      csv<<current_time<<",";
      csv<<current_time<<"/"<<phi<<"_"<<theta<<"_"<<m_image_name<<".png\n";
    }

    m_csv = csv.str();
    std::ofstream out(m_db_path + "/data.csv");
    out<<m_csv;
    out.close();

  }

private:
  void create_angles_from_range(std::vector<float> &angles, const float min,
                                const float inc, const int n_angles)
  {
    for(int a = 0; a < n_angles; ++a)
    {
      float angle =  float(min + inc * double(a));
      angles.push_back(angle);

    } // angles
  }

  void create_cinema_angles()
  {
    for(int p = 0; p < m_phi_values.size(); ++p)
    {
      for(int t = 0; t < m_theta_values.size(); ++t)
      {
        std::tuple<float,float> angles(m_phi_values[p], m_theta_values[t]);
        m_camera_angles.push_back(angles);

      } // theta
    } // phi
  }

  void create_cinema_cameras(vtkm::Bounds bounds)
  {
    m_cameras.clear();
    m_image_names.clear();
    using vtkmVec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkmVec3f center = bounds.Center();
    vtkm::Vec<vtkm::Float32,3> totalExtent;
    totalExtent[0] = vtkm::Float32(bounds.X.Length());
    totalExtent[1] = vtkm::Float32(bounds.Y.Length());
    totalExtent[2] = vtkm::Float32(bounds.Z.Length());

    vtkm::Float32 radius = vtkm::Magnitude(totalExtent) * 2.5 / 2.0;

    for(int a = 0; a < m_camera_angles.size(); ++a)
    {
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

      const float phi = std::get<0>(m_camera_angles[a]);
      const float theta = std::get<1>(m_camera_angles[a]);

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

      std::stringstream ss;
      ss<<get_string(phi)<<"_"<<get_string(theta)<<"_";

      m_image_names.push_back(ss.str() + m_image_name);
      m_cameras.push_back(camera);

    } // angles
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
                        const conduit::Node &render_node,
                        std::string db_name,
                        std::string path)
  {
    if(db_exists(db_name))
    {
      ASCENT_ERROR("Creation failed: cinema database already exists");
    }

    m_databases.emplace(std::make_pair(db_name, CinemaManager(bounds, render_node, db_name, path)));
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

    vtkm::Bounds *bounds = input<vtkm::Bounds>(0);

    std::vector<vtkh::Render> *renders = new std::vector<vtkh::Render>();

    Node meta = Metadata::n_metadata;

    int cycle = 0;

    if(meta.has_path("cycle"))
    {
      cycle = meta["cycle"].to_int32();
    }

    // figure out if we need the original bounds for the scene
    bool needs_original_bounds = false;
    if(params().has_path("renders"))
    {
      const conduit::Node renders_node = params()["renders"];
      const int num_renders = renders_node.number_of_children();

      for(int i = 0; i < num_renders; ++i)
      {
        const conduit::Node &render_node = renders_node.child(i);
        if(render_node.has_path("use_original_bounds"))
        {
          if(render_node["use_original_bounds"].as_string() == "true")
          {
            needs_original_bounds = true;
            break;
          }
        }
      }
    }
    else
    {
      if(params().has_path("use_original_bounds"))
      {
        if(params()["use_original_bounds"].as_string() == "true")
        {
          needs_original_bounds = true;
        }
      }
    }

    vtkm::Bounds original_bounds;
    if(needs_original_bounds)
    {
      DataObject *source
        = graph().workspace().registry().fetch<DataObject>("source_object");
      original_bounds = source->as_vtkh_collection()->global_bounds();
    }


    if(params().has_path("renders"))
    {
      const conduit::Node &renders_node = params()["renders"];
      const int num_renders = renders_node.number_of_children();

      for(int i = 0; i < num_renders; ++i)
      {
        const conduit::Node &render_node = renders_node.child(i);
        vtkm::Bounds scene_bounds = *bounds;
        if(render_node.has_path("use_original_bounds"))
        {
          if(render_node["use_original_bounds"].as_string() == "true")
          {
            scene_bounds = original_bounds;
          }
        }

        std::string image_name;

        bool is_cinema = false;
        bool is_auto_camera = false;

        if(render_node.has_path("type"))
        {
          if(render_node["type"].as_string() == "cinema")
          {
            is_cinema = true;
          }
          if(render_node["type"].as_string() == "auto_camera")
          {
            is_auto_camera = true;
          }
        }

        if(is_cinema)
        {
          if(!render_node.has_path("db_name"))
          {
            ASCENT_ERROR("Cinema must specify a 'db_name'");
          }

          std::string output_path = default_dir();

	  if(render_node.has_path("output_path"))
	  {
            output_path = render_node["output_path"].as_string();
	    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
            MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
            MPI_Comm_rank(mpi_comm, &rank);
#endif
            // create a folder if it doesn't exist
            if(rank == 0 && !conduit::utils::is_directory(output_path))
            {
              conduit::utils::create_directory(output_path);
            }
	  }

          if(!render_node.has_path("db_name"))
          {
            ASCENT_ERROR("Cinema must specify a 'db_name'");
          }
          std::string db_name = render_node["db_name"].as_string();
          bool exists = detail::CinemaDatabases::db_exists(db_name);
          if(!exists)
          {
            detail::CinemaDatabases::create_db(*bounds,render_node, db_name, output_path);
          }

          detail::CinemaManager &manager = detail::CinemaDatabases::get_db(db_name);
          // add this to the extract results in the registry
          if(!graph().workspace().registry().has_entry("extract_list"))
          {
            conduit::Node *extract_list = new conduit::Node();
            graph().workspace().registry().add<Node>("extract_list",
                                               extract_list,
                                               -1); // TODO keep forever?
          }

          conduit::Node *extract_list = graph().workspace().registry().fetch<Node>("extract_list");
          Node &einfo = extract_list->append();
          einfo["type"] = "cinema";
          einfo["path"] = manager.db_path();

          int image_width;
          int image_height;
          parse_image_dims(render_node, image_width, image_height);

          manager.set_bounds(scene_bounds);
          manager.add_time_step();
          manager.fill_renders(renders, render_node);
          manager.write_metadata();
        }
        else
        {
          // this render has a unique name
          if(render_node.has_path("image_name"))
          {
            image_name = render_node["image_name"].as_string();
            image_name = output_dir(image_name);
          }
          else if(render_node.has_path("image_prefix"))
          {
            std::stringstream ss;
            ss<<expand_family_name(render_node["image_prefix"].as_string(), cycle);
            image_name = ss.str();
            image_name = output_dir(image_name);
          }
          else
          {
            std::string render_name = renders_node.child_names()[i];
            std::string fpath = filter_to_path(this->name());
            ASCENT_ERROR("Render ("<<fpath<<"/"<<render_name<<")"<<
                         " must have either a 'image_name' or "
                         "'image_prefix' parameter");
          }


	  if(render_node.has_path("dataset_bounds"))
	  {
	    float64_accessor d_bounds = render_node["dataset_bounds"].value();
	    int num_bounds = d_bounds.number_of_elements();

	    if(num_bounds != 6)
            {
              std::string render_name = renders_node.child_names()[i];
              std::string fpath = filter_to_path(this->name());
              ASCENT_ERROR("Render ("<<fpath<<"/"<<render_name<<")"<<
                           " only provided " << num_bounds <<
	                   " dataset_bounds when 6 are required:" <<
			   " [xMin,xMax,yMin,yMax,zMin,zMax]");
	    }
	    if(scene_bounds.X.Min > d_bounds[0])
	      scene_bounds.X.Min = d_bounds[0];
	    if(scene_bounds.X.Max < d_bounds[1])
	      scene_bounds.X.Max = d_bounds[1];
	    if(scene_bounds.Y.Min > d_bounds[2])
	      scene_bounds.Y.Min = d_bounds[2];
	    if(scene_bounds.Y.Max < d_bounds[3])
	      scene_bounds.Y.Max = d_bounds[3];
	    if(scene_bounds.Z.Min > d_bounds[4])
	      scene_bounds.Z.Min = d_bounds[4];
	    if(scene_bounds.Z.Max < d_bounds[5])
	      scene_bounds.Z.Max = d_bounds[5];
	  }

	  if(is_auto_camera)
	  {
            DataObject *source
              = graph().workspace().registry().fetch<DataObject>("source_object");

            std::shared_ptr<VTKHCollection> collection = source->as_vtkh_collection();

	    if(!render_node.has_path("auto_camera/field"))
              ASCENT_ERROR("Auto Camera must specify a 'field'");
	    if(!render_node.has_path("auto_camera/metric"))
              ASCENT_ERROR("Auto Camera must specify a 'metric'");
	    if(!render_node.has_path("auto_camera/samples"))
              ASCENT_ERROR("Auto Camera must specify number of 'samples'");

            std::string field_name = render_node["auto_camera/field"].as_string();
            std::string metric     = render_node["auto_camera/metric"].as_string();
            int samples            = render_node["auto_camera/samples"].as_int32();

            if(!collection->has_field(field_name))
            {
              ASCENT_ERROR("Unknown field '"<<field_name<<"' in Auto Camera");
            }

            std::string topo_name = collection->field_topology(field_name);
            vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);

            vtkh::AutoCamera auto_cam;

	    int height = 1024;
	    int width  = 1024;
            if(render_node.has_path("auto_camera/bins"))
            {
              int bins = render_node["auto_camera/bins"].as_int32();
              auto_cam.SetNumBins(bins);
            }
            if(render_node.has_path("auto_camera/height"))
            {
              height = render_node["auto_camera/height"].as_int32();
              auto_cam.SetHeight(height);
            }
            if(render_node.has_path("auto_camera/width"))
            {
              width = render_node["auto_camera/width"].as_int32();
              auto_cam.SetWidth(width);
            }

            auto_cam.SetInput(&dataset);
            auto_cam.SetField(field_name);
            auto_cam.SetMetric(metric);
            auto_cam.SetNumSamples(samples);
            auto_cam.Update();

            vtkm::rendering::Camera *camera = new vtkm::rendering::Camera;
            *camera = auto_cam.GetCamera();
	    vtkh::Render render = vtkh::MakeRender(width,
                                      height,
                                      scene_bounds,
	    			      *camera,
                                      image_name);
            renders->push_back(render);
	    delete camera;

	  }
	  else
	  {

            vtkh::Render render = detail::parse_render(render_node,
                                                       scene_bounds,
                                                       image_name);
            renders->push_back(render);
	  }
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
        image_name = output_dir(image_name);
      }

      vtkm::Bounds scene_bounds = *bounds;
      if(params().has_path("use_original_bounds"))
      {
        if(params()["use_original_bounds"].as_string() == "true")
        {
          scene_bounds = original_bounds;
        }
      }

      vtkh::Render render = vtkh::MakeRender(1024,
                            1024,
                            scene_bounds,
                            image_name);

      Node meta = Metadata::n_metadata;
      if(meta.has_path("comments"))
      {
        const conduit::Node comments_node = meta["comments"];
        const int num_comments = comments_node.number_of_children();
        std::vector<std::string> comments;
        for(int i = 0; i < num_comments; ++i)
        {
          comments.push_back(comments_node.child(i).to_yaml());
        }
        render.SetComments(comments);
      }

      renders->push_back(render);
    }
    set_output<std::vector<vtkh::Render>>(renders);
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

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHBounds input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);

    // data could be the result of a failed pipeline
    if(data_object->is_valid())
    {
      std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
      bounds->Include(collection->global_bounds());
    }

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

    // this plot might have been created from a failed pipeline
    // so check to see if this is valid and only pass it on if
    // it is
    if(cont->is_valid())
    {
      scene->AddRenderer(cont);
    }
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

    res &= check_string("topology",params, info, false);
    valid_paths.push_back("topology");

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
      valid_paths.push_back("samples");
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
    if(!input(0).check_type<DataObject>())
    {
      ASCENT_ERROR("create_plot input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);


    if(!data_object->is_valid())
    {
      // this is trying to render a failed pipeline
      detail::RendererContainer *container = new detail::RendererContainer();
      set_output<detail::RendererContainer>(container);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    conduit::Node &plot_params = params();
    std::string field_name;
    if(plot_params.has_path("field"))
    {
      field_name = plot_params["field"].as_string();
    }
    std::string topo_name;
    if(field_name == "")
    {
      bool throw_error = false;
      topo_name = detail::resolve_topology(params(),
                                           this->name(),
                                           collection,
                                           throw_error);
      // don't crash everything, just warn the user and continue
      detail::RendererContainer *container = new detail::RendererContainer();
      set_output<detail::RendererContainer>(container);
    }
    else
    {
      topo_name = collection->field_topology(field_name);
      if(topo_name == "")
      {
        bool throw_error = false;
        detail::field_error(field_name, this->name(), collection, throw_error);
        // don't crash everything, just warn the user and continue
        detail::RendererContainer *container = new detail::RendererContainer();
        set_output<detail::RendererContainer>(container);
        return;
      }
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    std::string type = params()["type"].as_string();

    if(data.GlobalIsEmpty())
    {
      std::string fpath = filter_to_path(this->name());
      ASCENT_INFO(fpath<<" "<<type<<" plot yielded no data, i.e., no cells remain");
    }

    vtkh::Renderer *renderer = nullptr;

    if(type == "pseudocolor")
    {
      bool is_point_mesh = data.IsPointMesh();
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
      vtkh::VolumeRenderer *vren = new vtkh::VolumeRenderer();
      if(plot_params.has_path("samples"))
      {
        int samples = plot_params["samples"].to_int32();
        vren->SetNumberOfSamples(samples);
      }
      renderer = vren;
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
      vtkm::cont::ColorTable color_table = parse_color_table(plot_params["color_table"]);
      if(type != "mesh")
      {
        if(plot_params["color_table"].has_path("annotation"))
        {
          if(plot_params["color_table/annotation"].as_string() == "false")
          {
            renderer->DisableColorBar();
          }
        }
        if(type != "volume")
        {
          if(plot_params["color_table"].has_path("discrete"))
          {
            if(plot_params["color_table/discrete"].as_string() == "true")
            {
              renderer->SetDiscrete();
            }
          }
        }
      }
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

    if(field_name != "")
    {
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
        data.AddConstantPointField(0.f, fname);
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

    detail::RendererContainer *container
      = new detail::RendererContainer(key,
                                      &graph().workspace().registry(),
                                      renderer,
                                      collection,
                                      topo_name,
                                      *data_object);

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
void generate_camera_meshes(conduit::Node &image_data)
{
  conduit::Node &camera = image_data["camera"];

  if(camera.has_child("2d"))
  {
    // skip cam mesh for 2d cam mode
    return;
  }

  conduit::Node &cam_frust = camera["camera_frustum_mesh"];
  // std::string image_name = image_data["image_name"].as_string();

  // Scene Bounds Mesh
  float64_accessor scene_bounds = image_data["scene_bounds"].value();
  double x_scene_bounds[] = {scene_bounds[0], scene_bounds[0], scene_bounds[0], scene_bounds[0],
                             scene_bounds[3], scene_bounds[3], scene_bounds[3], scene_bounds[3]};
  double y_scene_bounds[] = {scene_bounds[1], scene_bounds[1], scene_bounds[4], scene_bounds[4],
                             scene_bounds[1], scene_bounds[1], scene_bounds[4], scene_bounds[4]};
  double z_scene_bounds[] = {scene_bounds[2], scene_bounds[5], scene_bounds[5], scene_bounds[2],
                             scene_bounds[2], scene_bounds[5], scene_bounds[5], scene_bounds[2]};

  cam_frust["coordsets/scene_bounds_coords/type"] = "explicit";
  cam_frust["coordsets/scene_bounds_coords/values/x"].set(x_scene_bounds, 8);
  cam_frust["coordsets/scene_bounds_coords/values/y"].set(y_scene_bounds, 8);
  cam_frust["coordsets/scene_bounds_coords/values/z"].set(z_scene_bounds, 8);

  cam_frust["topologies/scene_bounds_topo/type"] = "unstructured";
  cam_frust["topologies/scene_bounds_topo/coordset"] = "scene_bounds_coords";
  cam_frust["topologies/scene_bounds_topo/elements/shape"]  = "line";
  cam_frust["topologies/scene_bounds_topo/elements/connectivity"] = {0,1,1,2,2,3,3,0,
                                                                     4,5,5,6,6,7,7,4,
                                                                     0,4,1,5,2,6,3,7};

  // Initializing look vector from position to the "look_at" point of interest
  float64_accessor position = camera["position"].value();
  float64_accessor look_at = camera["look_at"].value();
  vtkm::Vec<vtkm::Float64,3> vtkm_look_at(look_at[0], look_at[1], look_at[2]);
  vtkm::Vec<vtkm::Float64,3> vtkm_position(position[0], position[1], position[2]);
  vtkm::Vec<vtkm::Float64,3> vtkm_look = vtkm_look_at - vtkm_position;
  vtkm::Normalize(vtkm_look);

  // Initializing and normalizing up vector
  float64_accessor up = camera["up"].value();
  vtkm::Vec<vtkm::Float64,3> vtkm_up(up[0], up[1], up[2]);

  vtkm::Vec<vtkm::Float64,3> forward(0,0,-1);
  double angle_between = vtkm::ACos(vtkm::Dot(forward, vtkm_look)) / vtkm::Pi() * 180;

  // If the look vector has been rotated by a certain angle, adjust the camera up vector to match
  if (vtkm::Abs(angle_between) >= 0.001) {
    vtkm::Vec<vtkm::Float64,3> axisOfRotation = vtkm::Cross(vtkm_look, forward);
    vtkm_up =
      vtkm::Transform3DVector(vtkm::Transform3DRotate(-angle_between, axisOfRotation), vtkm_up);
  }
  vtkm::Normalize(vtkm_up);

  // Identifying points where the look vector intersects with the near and far frustum planes
  double near_dist = camera["near_plane"].to_value();
  double far_dist = camera["far_plane"].to_value();
  vtkm::Vec<vtkm::Float64,3> vtkm_side = vtkm::Cross(vtkm_up, vtkm_look);
  vtkm::Vec<vtkm::Float64,3> look_near_pt = (vtkm_look * near_dist) + vtkm_position;
  vtkm::Vec<vtkm::Float64,3> look_far_pt = (vtkm_look * far_dist) + vtkm_position;

  // Calculating the bounds of the camera frustums
  int image_height = image_data["image_height"].to_value();
  int image_width = image_data["image_width"].to_value();
  double image_aspect = image_height/image_width;
  double fov = camera["fov"].to_value();
  double zoom = camera["zoom"].to_value();
  // Near frustum
  double frust_near_height = near_dist * vtkm::Tan(fov * 0.5 * vtkm::Pi() / 180.0) / zoom;
  double frust_near_width  = frust_near_height;
  vtkm::Vec<vtkm::Float64,3> near_frust_ll = look_near_pt + (-1 * vtkm_up * frust_near_height)
                                             + (vtkm_side * frust_near_width * image_aspect );
  vtkm::Vec<vtkm::Float64,3> near_frust_lr = look_near_pt + (-1 * vtkm_up * frust_near_height)
                                             + (-1 * vtkm_side * frust_near_width * image_aspect);
  vtkm::Vec<vtkm::Float64,3> near_frust_ur = look_near_pt + (vtkm_up * frust_near_height)
                                             + (-1 * vtkm_side * frust_near_width * image_aspect);
  vtkm::Vec<vtkm::Float64,3> near_frust_ul = look_near_pt + (vtkm_up * frust_near_height)
                                             + (vtkm_side * frust_near_width * image_aspect);
  // Far frustum
  double frust_far_height = far_dist * vtkm::Tan(fov * 0.5 * vtkm::Pi() / 180.0) / zoom;
  double frust_far_width  = frust_far_height;
  vtkm::Vec<vtkm::Float64,3> far_frust_ll = look_far_pt + (-1 * vtkm_up * frust_far_height)
                                            + (vtkm_side * frust_far_width * image_aspect);
  vtkm::Vec<vtkm::Float64,3> far_frust_lr = look_far_pt + (-1 * vtkm_up * frust_far_height)
                                            + (-1 * vtkm_side * frust_far_width * image_aspect);
  vtkm::Vec<vtkm::Float64,3> far_frust_ur = look_far_pt + (vtkm_up * frust_far_height)
                                            + (-1 * vtkm_side * frust_far_width * image_aspect);
  vtkm::Vec<vtkm::Float64,3> far_frust_ul = look_far_pt + (vtkm_up * frust_far_height)
                                            + (vtkm_side * frust_far_width * image_aspect);

  // Assembling frustum mesh
  vtkm::Vec<vtkm::Float64,3> up_vector_pt = vtkm_up * (far_dist - near_dist) * 0.5 + look_near_pt;
  double x_val_frust[] = {near_frust_ll[0],near_frust_lr[0],near_frust_ur[0],near_frust_ul[0],
                          far_frust_ll[0], far_frust_lr[0], far_frust_ur[0], far_frust_ul[0],
                          look_near_pt[0],  look_far_pt[0], up_vector_pt[0]};
  double y_val_frust[] = {near_frust_ll[1],near_frust_lr[1],near_frust_ur[1],near_frust_ul[1],
                          far_frust_ll[1], far_frust_lr[1], far_frust_ur[1], far_frust_ul[1],
                          look_near_pt[1],  look_far_pt[1], up_vector_pt[1]};
  double z_val_frust[] = {near_frust_ll[2],near_frust_lr[2],near_frust_ur[2],near_frust_ul[2],
                          far_frust_ll[2], far_frust_lr[2], far_frust_ur[2], far_frust_ul[2],
                          look_near_pt[2],  look_far_pt[2], up_vector_pt[2]};

  cam_frust["coordsets/camera_frustum_coords/type"] = "explicit";
  cam_frust["coordsets/camera_frustum_coords/values/x"].set(x_val_frust, 11);
  cam_frust["coordsets/camera_frustum_coords/values/y"].set(y_val_frust, 11);
  cam_frust["coordsets/camera_frustum_coords/values/z"].set(z_val_frust, 11);

  cam_frust["topologies/camera_frustum_topo/type"] = "unstructured";
  cam_frust["topologies/camera_frustum_topo/coordset"] = "camera_frustum_coords";
  cam_frust["topologies/camera_frustum_topo/elements/shape"]  = "line";
  cam_frust["topologies/camera_frustum_topo/elements/connectivity"] = {0,1,1,2,2,3,3,0, // Near plane
                                                                       4,5,5,6,6,7,7,4, // Far plane
                                                                       0,4,1,5,2,6,3,7, // Connect
                                                                       8,9,8,10}; // Look Vector

  cam_frust["topologies/clipping_planes_topo/type"] = "unstructured";
  cam_frust["topologies/clipping_planes_topo/coordset"] = "camera_frustum_coords";
  cam_frust["topologies/clipping_planes_topo/elements/shape"]  = "quad";
  cam_frust["topologies/clipping_planes_topo/elements/connectivity"] = {0,1,2,3,  // Near plane
                                                                       4,5,6,7,  // Far plane
                                                                       0,1,5,4,  // Lower plane
                                                                       2,3,7,6,  // Upper plane
                                                                       0,3,7,4,  // Left plane
                                                                       1,2,6,5}; // Right plane
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

      // check for 2d vs 3d camera
      if(renders->at(i).GetCamera().GetMode() ==  vtkm::rendering::Camera::Mode::TwoD)
      {
        vtkm::Bounds bounds =  renders->at(i).GetCamera().GetViewRange2D();
        double view_2d[4] = {bounds.X.Min,
                             bounds.Y.Min,
                             bounds.X.Max,
                             bounds.Y.Max};

        image_data["camera/2d"].set(view_2d,4);
      }
      else
      {
        image_data["camera/position"].set(&renders->at(i).GetCamera().GetPosition()[0],3);
        image_data["camera/look_at"].set(&renders->at(i).GetCamera().GetLookAt()[0],3);
        image_data["camera/up"].set(&renders->at(i).GetCamera().GetViewUp()[0],3);
        image_data["camera/zoom"] = renders->at(i).GetCamera().GetZoom();
        image_data["camera/fov"] = renders->at(i).GetCamera().GetFieldOfView();
        image_data["camera/near_plane"] = renders->at(i).GetCamera().GetClippingRange().Min;
        image_data["camera/far_plane"] = renders->at(i).GetCamera().GetClippingRange().Max;
      }
      auto pan_vals = renders->at(i).GetCamera().GetPan();
      image_data["camera/xpan"] = pan_vals[0];
      image_data["camera/ypan"] = pan_vals[1];
      vtkm::Bounds bounds=  renders->at(i).GetSceneBounds();
      double coord_bounds [6] = {bounds.X.Min,
                                 bounds.Y.Min,
                                 bounds.Z.Min,
                                 bounds.X.Max,
                                 bounds.Y.Max,
                                 bounds.Z.Max};

      image_data["scene_bounds"].set(coord_bounds, 6);

      generate_camera_meshes(image_data);
      image_list->append() = image_data;
    }

}
//-----------------------------------------------------------------------------

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
