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

#include "ascent_runtime_conduit_to_vtkm_parsing.hpp"

#include <ascent_logging.hpp>
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

double zoom_to_vtkm_zoom(double in_zoom)
{
  // vtkm is weird. increasing the value of zoom, zooms out.
  // we dont want that, so we have to convert what normal
  // people think of zoom into what vtkm wants.
  // vtkm zoom factor = pow(4.0, zoom)
  // log4 factor = vtkm_zoom
  // Ascent't input expects a zoom factor, ie 1= nozoom
  double vtkm_zoom = log(in_zoom) / log(4.0);
  return vtkm_zoom;
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
parse_camera(const conduit::Node camera_node, vtkm::rendering::Camera &camera)
{
  typedef vtkm::Vec<vtkm::Float32,3> vtkmVec3f;
  //
  // Get the optional camera parameters
  //
  if(camera_node.has_child("look_at"))
  {
      conduit::Node n;
      camera_node["look_at"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      vtkmVec3f look_at(coords[0], coords[1], coords[2]);
      camera.SetLookAt(look_at);
  }

  if(camera_node.has_child("position"))
  {
      conduit::Node n;
      camera_node["position"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
      vtkmVec3f position(coords[0], coords[1], coords[2]);
      camera.SetPosition(position);
  }

  if(camera_node.has_child("up"))
  {
      conduit::Node n;
      camera_node["up"].to_float64_array(n);
      const float64 *coords = n.as_float64_ptr();
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
      double zoom = camera_node["zoom"].to_float64();
      camera.Zoom(zoom_to_vtkm_zoom(zoom));
  }
  //
  // With a new potential camera position. We need to reset the
  // clipping plane as not to cut out part of the data set
  //

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

  // this is an offset from the current azimuth
  if(camera_node.has_child("azimuth"))
  {
      vtkm::Float64 azimuth = camera_node["azimuth"].to_float64();
      camera.Azimuth(azimuth);
  }
  if(camera_node.has_child("elevation"))
  {
      vtkm::Float64 elevation = camera_node["elevation"].to_float64();
      camera.Elevation(elevation);
  }
}

bool is_valid_name(const std::string &name)
{
  std::string lower_name;

  for(std::string::size_type i = 0; i < name.length(); ++i)
  {
    lower_name += std::tolower(name[i]);
  }

  std::set<std::string> presets = vtkm::cont::ColorTable::GetPresets();
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
vtkm::cont::ColorTable
parse_color_table(const conduit::Node &color_table_node)
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
            ASCENT_WARN("Unknown color table control point type "
                        << peg["type"].as_string()<<
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





