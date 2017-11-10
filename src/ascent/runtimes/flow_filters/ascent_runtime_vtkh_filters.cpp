//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_vtkh_data_adapter.hpp>

#endif

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

//-----------------------------------------------------------------------------
void 
parse_camera(const conduit::Node camera_node, vtkm::rendering::Camera &camera)
{
    typedef vtkm::Vec<vtkm::Float32,3> vtkmVec3f;
    //
    // Get the optional camera parameters
    //
    if(camera_node.has_child("look_at"))
    {
        const float64 *coords = camera_node["look_at"].as_float64_ptr();
        vtkmVec3f look_at(coords[0], coords[1], coords[2]);
        camera.SetLookAt(look_at);  
    }

    if(camera_node.has_child("position"))
    {
        const float64 *coords = camera_node["position"].as_float64_ptr();
        vtkmVec3f position(coords[0], coords[1], coords[2]);
        camera.SetPosition(position);  
    }
    
    if(camera_node.has_child("up"))
    {
        const float64 *coords = camera_node["up"].as_float64_ptr();
        vtkmVec3f up(coords[0], coords[1], coords[2]);
        vtkm::Normalize(up);
        camera.SetViewUp(up);
    }
    
    if(camera_node.has_child("fov"))
    {
        camera.SetFieldOfView(camera_node["fov"].to_float64());
    }

    if(camera_node.has_child("xpan") || camera_node.has_child("ypan"))
    {
        vtkm::Float64 xpan = 0.;
        vtkm::Float64 ypan = 0.;
        if(camera_node.has_child("xpan")) xpan = camera_node["xpan"].to_float64();
        if(camera_node.has_child("ypan")) xpan = camera_node["ypan"].to_float64();
        camera.Pan(xpan, ypan);
    }

    if(camera_node.has_child("zoom"))
    {
        camera.Zoom(camera_node["zoom"].to_float64());
    }
    //
    // With a new potential camera position. We need to reset the
    // clipping plane as not to cut out part of the data set
    //
    //this->SetDefaultClippingPlane();
    
    if(camera_node.has_child("near_plane"))
    {
        vtkm::Range clipping_range = camera.GetClippingRange();
        clipping_range.Min = camera_node["near_plane"].to_float64();
        camera.SetClippingRange(clipping_range);
    }

    if(camera_node.has_child("far_plane"))
    {
        vtkm::Range clipping_range = camera.GetClippingRange();
        clipping_range.Max = camera_node["far_plane"].to_float64();
        camera.SetClippingRange(clipping_range);
    }
}

vtkm::rendering::ColorTable 
parse_color_table(const conduit::Node &color_table_node)
{
  std::string color_map_name = "";
  if(color_table_node.has_child("name"))
  {
      color_map_name = color_table_node["name"].as_string();
  }

  vtkm::rendering::ColorTable color_table(color_map_name);

  if(color_map_name == "")
  {
      ASCENT_INFO("Color map name is empty. Ignoring");
      color_table.Clear();
  }
  
  if(!color_table_node.has_child("control_points"))
  {
      if(color_map_name == "") 
        ASCENT_ERROR("Error: a color table node was provided without a color map name or control points");
      return color_table;
  }
  
  NodeConstIterator itr = color_table_node.fetch("control_points").children();
  while(itr.has_next())
  {
      const Node &peg = itr.next();
      if(!peg.has_child("position"))
      {
          ASCENT_WARN("Color map control point must have a position. Ignoring");
      }
      float64 position = peg["position"].as_float64();
      
      if(position > 1.0 || position < 0.0)
      {
            ASCENT_WARN("Cannot add color map control point position "
                          << position 
                          << ". Must be a normalized scalar.");
      }

      if (peg["type"].as_string() == "rgb")
      {
          const float64 *color = peg["color"].as_float64_ptr();
          
          vtkm::rendering::Color ecolor(color[0], color[1], color[2]);
          
          color_table.AddControlPoint(position, ecolor);
      }
      else if (peg["type"].as_string() == "alpha")
      {
          float64 alpha = peg["alpha"].to_float64();
          color_table.AddAlphaControlPoint(position, alpha);
      }
      else
      {
          ASCENT_WARN("Unknown color table control point type " << peg["type"].as_string()<<
                      "\nValid types are 'alpha' and 'rgb'");
      }
  }

  return color_table;
}

vtkh::Render parse_render(const conduit::Node &render_node, 
                          vtkm::Bounds &bounds, 
                          const std::vector<vtkm::Id> &domain_ids,
                          const std::string &image_name)
{
  int image_width = 1024; 
  int image_height = 1024;

  if(render_node.has_path("image_width"))
  {
    image_width = render_node["image_width"].as_int32();
  }

  if(render_node.has_path("image_height"))
  {
    image_height = render_node["image_height"].as_int32();
  }
  
  //
  // for now, all the canvases we support are the same
  // so passing MakeRender a RayTracer is ok
  //
  vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(image_width,
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

  if(render_node.has_path("color_table"))
  {
    vtkm::rendering::ColorTable color_table =  parse_color_table(render_node["color_table"]);
    render.SetColorTable(color_table);
  }
  return render;
}

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
    if(input(0).check_type<Node>())
    {
        // convert from blueprint to vtk-h
        const Node *n_input = input<Node>(0);
        vtkh::DataSet *res = VTKHDataAdapter::BlueprintToVTKHDataSet(*n_input);
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


EnsureBlueprint::EnsureBlueprint()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
EnsureBlueprint::~EnsureBlueprint()
{
// empty
}

//-----------------------------------------------------------------------------
void 
EnsureBlueprint::declare_interface(Node &i)
{
    i["type_name"]   = "ensure_blueprint";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void 
EnsureBlueprint::execute()
{
    if(input(0).check_type<vtkh::DataSet>())
    {
        // convert from vtk-h to blueprint
        vtkh::DataSet *in_dset = input<vtkh::DataSet>(0);
        const vtkm::Id num_domains = in_dset->GetNumberOfDomains();
        conduit::Node * res = new conduit::Node();
        uint64 cycle = in_dset->GetCycle();
        for(vtkm::Id dom = 0; dom < num_domains; ++dom)
        {
            vtkm::cont::DataSet dset; 
            vtkm::Id domain_id;
            in_dset->GetDomain(dom, dset, domain_id);
            dset.PrintSummary(std::cout);
            conduit::Node &bp = res->append();
            VTKHDataAdapter::VTKmToBlueprintDataSet(&dset, bp);
            bp["state/cycle"] = cycle;
        }

        set_output<conduit::Node>(res);
    }
    else if(input(0).check_type<vtkm::cont::DataSet>())
    {
        // wrap our vtk-m dataset in vtk-h
        conduit::Node *res = new conduit::Node();
        VTKHDataAdapter::VTKmToBlueprintDataSet(input<vtkm::cont::DataSet>(0), *res);
        set_output<conduit::Node>(res);
    }
    else if(input(0).check_type<Node>())
    {
        // our data is already a node, pass though
        conduit::Node *res = input<Node>(0);
        conduit::Node info;
        bool success = conduit::blueprint::verify("mesh",*res,info);

        if(!success)
        {
          info.print();
          ASCENT_ERROR("conduit::Node input to EnsureBlueprint is non-conforming") 
        }

        set_output(input(0));
    }
    else
    {
        ASCENT_ERROR("ensure_blueprint input must be a data set"
                     "conforming conduit::Node, a vtk-m dataset, or vtk-h dataset");
    }
}


//-----------------------------------------------------------------------------
VTKHVolumeTracer::VTKHVolumeTracer()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVolumeTracer::~VTKHVolumeTracer()
{
// empty
}

//-----------------------------------------------------------------------------
void 
VTKHVolumeTracer::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_volumetracer";
    i["port_names"].append() = "in";
    i["port_names"].append() = "renders";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHVolumeTracer::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();   
    bool res = true;
    
    if(! params.has_child("field") || 
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
    }
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHVolumeTracer::execute()
{
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_volumetracer input0 must be a vtk-h dataset");
    }
    if(!input(1).check_type<std::vector<vtkh::Render>>())
    {
        ASCENT_ERROR("vtkh_volumeracer input1 must be a vth-h render");
    }
 
    ASCENT_INFO("Doing the render!");
    bool composite = true;
    //
    // there is no need to check for compositing param
    // since a volume plot will always be at the end of 
    // a series of plots
    //
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    std::vector<vtkh::Render> *renders = input<std::vector<vtkh::Render>>(1);
    vtkh::VolumeRenderer tracer;  
    tracer.SetInput(data);
    tracer.SetDoComposite(composite);
    tracer.SetRenders(*renders);
    tracer.SetField(params()["field"].as_string());
    tracer.Update();
    
    std::vector<vtkh::Render> out_renders = tracer.GetRenders();
    //
    // We need to create a new pointer for the output because the input will be deleted
    // There is only a small amount of overhead since the canvases contained 
    // in the render will be shallow copied.
    //
    std::vector<vtkh::Render> *renders_ptr = new std::vector<vtkh::Render>();
    *renders_ptr = out_renders;
    set_output<std::vector<vtkh::Render>>(renders_ptr);
}

VTKHRayTracer::VTKHRayTracer()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHRayTracer::~VTKHRayTracer()
{
// empty
}

//-----------------------------------------------------------------------------
void 
VTKHRayTracer::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_raytracer";
    i["port_names"].append() = "in";
    i["port_names"].append() = "renders";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHRayTracer::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();   
    bool res = true;
    
    if(! params.has_child("field") || 
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
    }
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHRayTracer::execute()
{
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_raytracer input0 must be a vtk-h dataset");
    }
    if(!input(1).check_type<std::vector<vtkh::Render>>())
    {
        ASCENT_ERROR("vtkh_raytracer input1 must be a vth-h render");
    }
 
    ASCENT_INFO("Doing the render!");
    bool composite = true;
    if(params().has_path("composite"))
    {
      if(params()["composite"].as_string() == "false")
      {
        composite = false; 
      }
    }    

    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    std::vector<vtkh::Render> *renders = input<std::vector<vtkh::Render>>(1);
    vtkh::RayTracer ray_tracer;  
    ray_tracer.SetInput(data);
    ray_tracer.SetDoComposite(composite);
    ray_tracer.SetRenders(*renders);
    ray_tracer.SetField(params()["field"].as_string());
    ray_tracer.Update();
    
    std::vector<vtkh::Render> out_renders = ray_tracer.GetRenders();
    //
    // We need to create a new pointer for the output because the input will be deleted
    // There is only a small amount of overhead since the canvases contained 
    // in the render will be shallow copied.
    //
    std::vector<vtkh::Render> *renders_ptr = new std::vector<vtkh::Render>();
    *renders_ptr = out_renders;
    set_output<std::vector<vtkh::Render>>(renders_ptr);
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
    bool res = true;
    
    if(! params.has_child("field") || 
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
    }
    
    if(! params.has_child("iso_values") || 
       ! params["iso_values"].dtype().is_number() )
    {
        info["errors"].append() = "Missing required numeric parameter 'iso_values'";
    }
    
    return res;
}

//-----------------------------------------------------------------------------
void 
VTKHMarchingCubes::execute()
{

    ASCENT_INFO("Marching the cubes!");
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("vtkh_marchingcubes input must be a vtk-h dataset");
    }

    std::string field_name = params()["field"].as_string();
    
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::MarchingCubes marcher;
    
    marcher.SetInput(data);
    marcher.SetField(field_name);

    const Node &n_iso_vals = params()["iso_values"];

    // convert to contig doubles
    Node n_iso_vals_dbls;
    n_iso_vals.to_float64_array(n_iso_vals_dbls);
    
    marcher.SetIsoValues(n_iso_vals.as_double_ptr(),
                         n_iso_vals.dtype().number_of_elements());

    marcher.Update();

    vtkh::DataSet *iso_output = marcher.GetOutput();
    
    set_output<vtkh::DataSet>(iso_output);
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
    bool res = true;
    
    if(! params.has_child("field") || 
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
    }
    
    if(! params.has_child("") || 
       ! params["min_value"].dtype().is_number() )
    {
        info["errors"].append() = "Missing required numeric parameter 'min_value'";
    }
    if(! params.has_child("") || 
       ! params["max_value"].dtype().is_number() )
    {
        info["errors"].append() = "Missing required numeric parameter 'max_value'";
    }
    
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHThreshold::execute()
{

    ASCENT_INFO("Thresholding!");
    
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
    double min_val = n_min_val.as_float64(); 
    double max_val = n_max_val.as_float64(); 
    thresher.SetUpperThreshold(max_val);
    thresher.SetLowerThreshold(min_val);

    thresher.AddMapField(field_name);
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
    bool res = true;
    if(! params.has_child("image_prefix") )
    {
        info["errors"].append() = "Missing required string parameter 'image_prefix'";
    }
    return res;
}

//-----------------------------------------------------------------------------

void 
DefaultRender::execute()
{

    ASCENT_INFO("We be default rendering!");
    
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

    if(params().has_path("renders"))
    {
      const conduit::Node renders_node = params()["renders"];
      const int num_renders= renders_node.number_of_children();
      for(int i = 0; i < num_renders; ++i)
      {
        const conduit::Node render_node = renders_node.child(i);
        std::string image_name;

        if(render_node.has_path("image_name"))
        {
          image_name = render_node["image_name"].as_string();
        }
        else
        {
          std::stringstream ss;
          ss<<params()["image_prefix"].as_string();
          ss<<"_"<<i;
          image_name = ss.str(); 
        }

        vtkh::Render render = detail::parse_render(render_node, 
                                                   *bounds, 
                                                   v_domain_ids, 
                                                   image_name);
        renders->push_back(render); 
      }
    }
    else
    {
      vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(1024,
                                                              1024, 
                                                              *bounds,
                                                              v_domain_ids,
                                                              params()["image_prefix"].as_string());

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
    
    if(! params.has_child("") || 
       ! params["sphere"].dtype().is_number() )
    {
        info["errors"].append() = "Missing required numeric parameter 'sphere'";
    }
    
    // TODO: check for other clip types 
    return res;
}


//-----------------------------------------------------------------------------
void 
VTKHClip::execute()
{

    ASCENT_INFO("We be clipping!");
    
    if(!input(0).check_type<vtkh::DataSet>())
    {
        ASCENT_ERROR("VTKHClip input must be a vtk-h dataset");
    }

    
    vtkh::DataSet *data = input<vtkh::DataSet>(0);
    vtkh::Clip clipper;
    
    clipper.SetInput(data);

    if(params().has_child("topology"))
    {
      std::string topology = params()["topology"].as_string();
      clipper.SetCellSet(topology);
    }

    const Node &sphere = params()["sphere"];
    double center[3];

    center[0] = sphere["center/x"].as_float64();
    center[1] = sphere["center/y"].as_float64();
    center[2] = sphere["center/z"].as_float64();
    double radius = sphere["radius"].as_float64(); 
  
    clipper.SetSphereClip(center, radius);
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
        // convert from conduit to vtkm
        const Node *n_input = input<Node>(0);
        vtkm::cont::DataSet  *res = VTKHDataAdapter::BlueprintToVTKmDataSet(*n_input);
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
    ASCENT_INFO("VTK-h bounds");
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
    result->Include(*bounds_a);
    
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
    ASCENT_INFO("VTK-h domain_ids");
    
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
int Scene::s_image_count = 0;

Scene::Scene()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Scene::~Scene()
{
// empty
}


//-----------------------------------------------------------------------------
bool
Scene::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = true;

    // TODO
    return res;
}

//-----------------------------------------------------------------------------
void 
Scene::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_scene";
    i["port_names"].append() = "bounds";
    i["port_names"].append() = "domain_ids";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void 
Scene::execute()
{
    ASCENT_INFO("Creating a scene default renderer!");
    
    // inputs are bounds and set of domains
    vtkm::Bounds       *bounds_in     = input<vtkm::Bounds>(0);
    std::set<vtkm::Id> *domain_ids_set = input<std::set<vtkm::Id> >(1);
    
    std::stringstream ss;
    ss<<"default_image_"<<s_image_count;
    s_image_count++;
    
    vtkm::Bounds bounds;
    bounds.Include(*bounds_in);
    
    std::vector<vtkm::Id> domain_ids(domain_ids_set->begin(),
                                     domain_ids_set->end());

    
    vtkh::Render render = vtkh::MakeRender<vtkh::RayTracer>(1024,
                                                            1024, 
                                                            bounds,
                                                            domain_ids,
                                                            ss.str());

    std::vector<vtkh::Render> *renders = new std::vector<vtkh::Render>();
    renders->push_back(render);
    set_output<std::vector<vtkh::Render> >(renders);
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





