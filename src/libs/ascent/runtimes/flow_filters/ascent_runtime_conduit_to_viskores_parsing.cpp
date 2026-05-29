//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_rover_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_conduit_to_viskores_parsing.hpp"

#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>
using namespace conduit;

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

bool string_equal(const std::string& str1, const std::string& str2)
{
  if (str1.size() != str2.size())
  {
    return false;
  }

  auto itr1 = str1.begin();
  auto itr2 = str2.begin();
  while ((itr1 != str1.end()) && (itr2 != str2.end()))
  {
    if (std::tolower(*itr1) != std::tolower(*itr2))
    {
      return false;
    }
    ++itr1;
    ++itr2;
  }

  return true;
}

double zoom_to_viskores_zoom(double in_zoom)
{
  // viskores is weird. increasing the value of zoom, zooms out.
  // we dont want that, so we have to convert what normal
  // people think of zoom into what viskores wants.
  // viskores zoom factor = pow(4.0, zoom)
  // log4 factor = viskores_zoom
  // Ascent't input expects a zoom factor, ie 1= nozoom
  double viskores_zoom = log(in_zoom) / log(4.0);
  return viskores_zoom;
}

void
parse_image_dims(const conduit::Node &node, int &width, int &height)
{
  width = 1024;
  height = 1024;

  if(node.has_path("image_width"))
  {
    width = node["image_width"].to_int32();
  }

  if(node.has_path("image_height"))
  {
    height = node["image_height"].to_int32();
  }

}


//-----------------------------------------------------------------------------
void
parse_camera(const conduit::Node camera_node, viskores::rendering::Camera &camera)
{
  typedef viskores::Vec<viskores::Float32,3> viskoresVec3f;

  //
  // Get the optional camera parameters
  //

  // -------------------- //
  // 2D camera parameters //
  // -------------------- //
  if(camera_node.has_child("2d") || camera_node.has_child("windowCoords"))
  {
    // camera:
    //  2d: [l,r,b,t]
    camera.SetModeTo2D();
    float64_accessor view_vals;
    if(camera_node.has_child("2d"))
    {
        view_vals = camera_node["2d"].value();
    }
    else
    {
        view_vals = camera_node["windowCoords"].value();
    }

    camera.SetViewRange2D(view_vals[0],
                          view_vals[1],
                          view_vals[2],
                          view_vals[3]);
  }

  // ------------------------ //
  // Ascent camera parameters //
  // ------------------------ //
  if(camera_node.has_child("look_at"))
  {
      conduit::Node n;
      camera_node["look_at"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      viskoresVec3f look_at(coords[0], coords[1], coords[2]);
      camera.SetLookAt(look_at);
  }

  if(camera_node.has_child("position"))
  {
      conduit::Node n;
      camera_node["position"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      viskoresVec3f position(coords[0], coords[1], coords[2]);
      camera.SetPosition(position);
  }

  if(camera_node.has_child("up"))
  {
      conduit::Node n;
      camera_node["up"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      viskoresVec3f up(coords[0], coords[1], coords[2]);
      viskores::Normalize(up);
      camera.SetViewUp(up);
  }

  if(camera_node.has_child("fov"))
  {
      camera.SetFieldOfView(camera_node["fov"].to_float64());
  }

  if(camera_node.has_child("xpan") || camera_node.has_child("ypan"))
  {
      viskores::Float64 xpan = 0.;
      viskores::Float64 ypan = 0.;
      if(camera_node.has_child("xpan")) xpan = camera_node["xpan"].to_float64();
      if(camera_node.has_child("ypan")) ypan = camera_node["ypan"].to_float64();
      camera.Pan(xpan, ypan);
  }

  if(camera_node.has_child("zoom"))
  {
      double zoom = camera_node["zoom"].to_float64();
      camera.Zoom(zoom_to_viskores_zoom(zoom));
  }

  //
  // With a new potential camera position. We need to reset the
  // clipping plane as not to cut out part of the data set
  //

  if(camera_node.has_child("near_plane"))
  {
      viskores::Range clipping_range = camera.GetClippingRange();
      clipping_range.Min = camera_node["near_plane"].to_float64();
      camera.SetClippingRange(clipping_range);
  }

  if(camera_node.has_child("far_plane"))
  {
      viskores::Range clipping_range = camera.GetClippingRange();
      clipping_range.Max = camera_node["far_plane"].to_float64();
      camera.SetClippingRange(clipping_range);
  }

  // this is an offset from the current azimuth
  if(camera_node.has_child("azimuth"))
  {
      viskores::Float64 azimuth = camera_node["azimuth"].to_float64();
      camera.Azimuth(azimuth);
  }

  if(camera_node.has_child("elevation"))
  {
      viskores::Float64 elevation = camera_node["elevation"].to_float64();
      camera.Elevation(elevation);
  }

  // ----------------------- //
  // Visit camera parameters //
  // ----------------------- //
  if(camera_node.has_child("focus"))
  {
      float64_accessor coords = camera_node["focus"].value();
      viskoresVec3f look_at(coords[0], coords[1], coords[2]);
      camera.SetLookAt(look_at);
  }

  if(camera_node.has_child("view_normal") && 
     camera_node.has_child("focus") && 
     camera_node.has_child("view_angle") && 
     camera_node.has_child("parallel_scale"))
  {
      // Compute camera distance using perspective projection formula
      conduit::float64 view_angle = camera_node["view_angle"].to_float64() * (viskores::Pi() / 360.0);
      conduit::float64 parallel_scale = camera_node["parallel_scale"].to_float64();
      conduit::float64 distance = (parallel_scale) / std::tan(view_angle);

      // Normalize viewNormal vector
      float64_accessor view = camera_node["view_normal"].value();
      conduit::float64 norm = std::sqrt(view[0]*view[0] + view[1]*view[1] + view[2]*view[2]);
      conduit::float64 view_norm[3] = { view[0]/norm, view[1]/norm, view[2]/norm };

      float64_accessor focus = camera_node["focus"].value();
      conduit::float64 pos[3];
      for(int i = 0; i < 3; ++i)
      {
        pos[i] = focus[i] + view_norm[i] * distance;
      }

      viskoresVec3f position(pos[0], pos[1], pos[2]);
      camera.SetPosition(position);

      // Clipping planes require positional information to be computed
      if(camera_node.has_child("near_plane"))
      {
        viskores::Range clipping_range = camera.GetClippingRange();
        clipping_range.Min =std::max(CONDUIT_EPSILON, distance + camera_node["near_plane"].to_float64());
        camera.SetClippingRange(clipping_range);
      }
      
      if(camera_node.has_child("far_plane"))
      {
        viskores::Range clipping_range = camera.GetClippingRange();
        clipping_range.Max = distance + camera_node["far_plane"].to_float64();
        camera.SetClippingRange(clipping_range);
      }
  }

  if(camera_node.has_child("view_angle"))
  {
      camera.SetFieldOfView(camera_node["view_angle"].to_float64() * 2);
  }

  if(camera_node.has_child("view_up"))
  {
      float64_accessor coords = camera_node["view_up"].value();
      viskoresVec3f up(coords[0], coords[1], coords[2]);
      viskores::Normalize(up);
      camera.SetViewUp(up);
  }

  if(camera_node.has_child("image_pan"))
  {
      float64_accessor pan_xy = camera_node["image_pan"].value();
      camera.Pan(pan_xy[0], pan_xy[1]);
  }

  if(camera_node.has_child("image_zoom"))
  {
      camera.Zoom(zoom_to_viskores_zoom(camera_node["image_zoom"].to_float64()));
  }

  if(camera_node.has_child("perspective"))
  {
    ASCENT_INFO("Visit camera parameter \"perspective\" is recognized but currently has no effect");
  }

  if(camera_node.has_child("eye_angle"))
  {
    ASCENT_INFO("Visit camera parameter \"eye_angle\" is recognized but currently has no effect");
  }

  if(camera_node.has_child("center_of_rotation_set") ||
     camera_node.has_child("center_of_rotation"))
  {
    ASCENT_INFO("Visit camera parameter \"center_of_rotation\" and \"center_of_rotation_set\" are recognized but currently have no effect");
  }

  if(camera_node.has_child("axis_3d_scale_flag") ||
     camera_node.has_child("axis_3d_scale"))
  {
    ASCENT_INFO("Visit camera parameter \"axis_3d_scale\" and \"axis_3d_scale_flag\" are recognized but currently have no effect");
  }

  if(camera_node.has_child("shear"))
  {
    ASCENT_INFO("Visit camera parameter \"shear\" is recognized but currently has no effect");
  }

  if(camera_node.has_child("window_valid"))
  {
    ASCENT_INFO("Visit camera parameter \"window_valid\" is recognized but currently has no effect");
  }
}

bool is_valid_name(const std::string &name)
{
  std::string lower_name;

  for(std::string::size_type i = 0; i < name.length(); ++i)
  {
    lower_name += std::tolower(name[i]);
  }

  std::set<std::string> presets = viskores::cont::ColorTable::GetPresets();
  bool valid = false;
  for( auto s : presets)
  {
    valid = string_equal(s, name);
    if(valid)
    {
      break;
    }
  }

  return valid;
}
//-----------------------------------------------------------------------------
viskores::cont::ColorTable
parse_color_table(const conduit::Node &color_table_node)
{
  // default name
  std::string color_map_name = "cool to warm";

  if(color_table_node.number_of_children() == 0)
  {
    ASCENT_INFO("Color table node is empty (no children). Defaulting to "
                <<color_map_name);
  }

  if(color_table_node.has_child("solid"))
  {
    viskores::cont::ColorTable color_table;
    color_table.ClearColors();

    float64_array color_vals = color_table_node.fetch("solid").value();
    viskores::Vec<viskores::Float64,3> ecolor(color_vals[0], color_vals[1], color_vals[2]);
    color_table.AddPoint(0.0, ecolor);

    if (color_vals.number_of_elements() == 4)
    {
      color_table.AddPointAlpha(0.0, std::min(1., std::max(color_vals[3], 0.)));
    }

    return color_table;
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

  viskores::cont::ColorTable color_table(color_map_name);

  if(color_table_node.has_child("control_points"))
  {
    const Node &control_points_node = color_table_node.fetch("control_points");

    // check to see if we have rgb points and clear the table
    bool clear = false;
    if (control_points_node.dtype().is_list())
    {
        NodeConstIterator itr = control_points_node.children();
        while(itr.has_next())
        {
            const Node &peg = itr.next();
            if (peg["type"].as_string() == "rgb")
            {
                clear = true;
                break;
            }
        }
    }
    else if (control_points_node.dtype().is_object())
    {
        if (control_points_node.has_child("r") &&
            control_points_node.has_child("g") &&
            control_points_node.has_child("b"))
        {
            clear = true;
        }
    }

    if(clear && !name_provided)
    {
      color_table.ClearColors();
    }

    if (control_points_node.dtype().is_list())
    {
        NodeConstIterator itr = control_points_node.children();
        while(itr.has_next())
        {
            const Node &peg = itr.next();
            if(!peg.has_child("position"))
            {
                // FIXME: This should be an error
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

                viskores::Vec<viskores::Float64,3> ecolor(color[0], color[1], color[2]);

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
                ASCENT_WARN("Unknown color table control point type "
                            << peg["type"].as_string()<<
                            "\nValid types are 'alpha' and 'rgb'");
            }
        }
    }
    else if (control_points_node.dtype().is_object())
    {
        if(!control_points_node.has_child("r"))
        {
            ASCENT_ERROR("Color map control point must provide r values");
        }

        if(!control_points_node.has_child("g"))
        {
            ASCENT_ERROR("Color map control point must provide g values");
        }

        if(!control_points_node.has_child("b"))
        {
            ASCENT_ERROR("Color map control point must provide b values");
        }

        if(!control_points_node.has_child("position"))
        {
            ASCENT_ERROR("Color map control point must have a position");
        }

        float64_array r_vals = control_points_node.fetch("r").value();
        float64_array g_vals = control_points_node.fetch("g").value();
        float64_array b_vals = control_points_node.fetch("b").value();
        float64_array pos_vals = control_points_node.fetch("position").value();

        if(r_vals.number_of_elements() != g_vals.number_of_elements() ||
            g_vals.number_of_elements() != b_vals.number_of_elements() ||
            b_vals.number_of_elements() != pos_vals.number_of_elements())
        {
            ASCENT_ERROR("Color map color channels should all be of the same size");
        }

        for(index_t i=0; i<r_vals.number_of_elements();i++)
        {
            viskores::Vec<viskores::Float64,3> ecolor(r_vals[i], g_vals[i], b_vals[i]);

            for(int i = 0; i < 3; ++i)
            {
                ecolor[i] = std::min(1., std::max(ecolor[i], 0.));
            }

            if(pos_vals[i] > 1.0 || pos_vals[i] < 0.0)
            {
                ASCENT_ERROR("Cannot add color map control point position "
                                << pos_vals[i]
                                << ". Must be a normalized scalar.");
            }

            color_table.AddPoint(pos_vals[i], ecolor);
        }

        if(control_points_node.has_child("a"))
        {
            float64_array alpha_vals = control_points_node.fetch("a").value();

            if(pos_vals.number_of_elements() != alpha_vals.number_of_elements())
            {
                ASCENT_ERROR("Color map alpha channel should have same size as color channels");
            }

            for(index_t i=0; i<alpha_vals.number_of_elements();i++)
            {
                color_table.AddPointAlpha(pos_vals[i], std::min(1., std::max(alpha_vals[i], 0.)));
            }
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





