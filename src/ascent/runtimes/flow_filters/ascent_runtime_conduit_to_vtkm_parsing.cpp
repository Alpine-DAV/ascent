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

void
parse_image_dims(const conduit::Node &node, int &width, int &height)
{
  width = 800;
  height = 800;

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
      camera.Zoom(camera_node["zoom"].to_float64());
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
  bool valid = false;
  if(lower_name == "default" ||
     lower_name == "cool to warm" ||
     lower_name == "cool to warm extended" ||
     lower_name == "viridis" ||
     lower_name == "inferno" ||
     lower_name == "plasma" ||
     lower_name == "black-body radiation" ||
     lower_name == "x ray" ||
     lower_name == "green" ||
     lower_name == "black - blue - white" ||
     lower_name == "blue to orange" ||
     lower_name == "gray to red" ||
     lower_name == "cool and hot" ||
     lower_name == "blue - green - orange" ||
     lower_name == "yellow - gray - blue" ||
     lower_name == "rainbow uniform" ||
     lower_name == "jet" ||
     lower_name == "rainbow desaturated")
  {
    valid = true;
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





