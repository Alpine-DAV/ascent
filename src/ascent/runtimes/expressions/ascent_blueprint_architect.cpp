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
/// file: ascent_blueprint_architect.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_blueprint_architect.hpp"

#include <ascent_logging.hpp>

#include <limits>

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
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

void logical_index_2d(int *idx, int point_index, int *dims)
{
  idx[0] = point_index % dims[0];
  idx[1] = point_index / dims[0];
}

void logical_index_3d(int *idx, int point_index, int *dims)
{
  idx[0] = point_index % dims[0];
  idx[1] = (point_index / dims[0]) % dims[1];
  idx[2] = point_index / (dims[0] * dims[1]);
}

conduit::Node
get_uniform_point(const conduit::Node &n_coords, const int &index)
{
  const conduit::Node &n_dims = n_coords["dims"];

  int dims[3] = {0,0,0};
  dims[0] = n_dims["i"].to_int();
  dims[1] = n_dims["j"].to_int();
  dims[2] = 1;

  bool is_2d = true;

  // check for 3d
  if(n_dims.has_path("k"))
  {
      dims[2] = n_dims["k"].to_int();
      is_2d = false;
  }


  conduit::float64 origin[3];
  conduit::float64 spacing[3];
  origin[0] = 0.0;
  origin[1] = 0.0;
  origin[2] = 0.0;

  spacing[0] = 1.0;
  spacing[1] = 1.0;
  spacing[2] = 1.0;

  const conduit::Node &n_origin = n_coords["origin"];

  origin[0] = n_origin["x"].to_float64();
  origin[1] = n_origin["y"].to_float64();

  if(n_origin.has_child("z"))
  {
    origin[2] = n_origin["z"].to_float64();
  }

  const conduit::Node &n_spacing = n_coords["spacing"];

  spacing[0] = n_spacing["dx"].to_float64();
  spacing[1] = n_spacing["dy"].to_float64();

  if(n_spacing.has_path("dz"))
  {
    spacing[2] = n_spacing["dz"].to_float64();
  }

  int logical_index[3] = {0, 0, 0};

  if(is_2d)
  {
    logical_index_2d(logical_index, index, dims);
  }
  else
  {
    logical_index_3d(logical_index, index, dims);
  }

  double point[3];
  point[0] = origin[0] + logical_index[0] * spacing[0];
  point[1] = origin[1] + logical_index[1] * spacing[1];
  point[2] = origin[2] + logical_index[2] * spacing[2];


  conduit::Node res;
  res.set(point,3);
  return res;
}

conduit::Node
get_explicit_point(const conduit::Node &n_coords, const int &index)
{
  bool is_float64 = true;
  if(n_coords["values/x"].dtype().is_float32())
  {
    is_float64 = false;
  }
  double point[3] = {0., 0., 0.};
  if(is_float64)
  {
    conduit::float64_array x_a = n_coords["values/x"].value();
    conduit::float64_array y_a = n_coords["values/y"].value();
    conduit::float64_array z_a = n_coords["values/z"].value();
    point[0] = x_a[index];
    point[1] = y_a[index];
    point[2] = z_a[index];
  }
  else
  {
    conduit::float32_array x_a = n_coords["values/x"].value();
    conduit::float32_array y_a = n_coords["values/y"].value();
    conduit::float32_array z_a = n_coords["values/z"].value();
    point[0] = x_a[index];
    point[1] = y_a[index];
    point[2] = z_a[index];

  }

  conduit::Node res;
  res.set(point,3);
  return res;
}

conduit::Node
get_rectilinear_point(const conduit::Node &n_coords, const int &index)
{
  bool is_float64 = true;

  int dims[3] = {0,0,0};
  dims[0] = n_coords["values/x"].dtype().number_of_elements();
  dims[1] = n_coords["values/y"].dtype().number_of_elements();

  if(n_coords.has_path("values/z"))
  {
    dims[2] = n_coords["values/z"].dtype().number_of_elements();
  }

  if(n_coords["values/x"].dtype().is_float32())
  {
    is_float64 = false;
  }
  double point[3] = {0., 0., 0.};


  int logical_index[3] = {0, 0, 0};

  if(dims[2] == 0)
  {
    logical_index_2d(logical_index, index, dims);
  }
  else
  {
    logical_index_3d(logical_index, index, dims);
  }

  if(is_float64)
  {
    conduit::float64_array x_a = n_coords["values/x"].value();
    conduit::float64_array y_a = n_coords["values/y"].value();
    point[0] = x_a[logical_index[0]];
    point[1] = y_a[logical_index[1]];
    if(dims[2] != 0)
    {
      conduit::float64_array z_a = n_coords["values/z"].value();
      point[2] = z_a[logical_index[2]];
    }
  }
  else
  {
    conduit::float32_array x_a = n_coords["values/x"].value();
    conduit::float32_array y_a = n_coords["values/y"].value();
    point[0] = x_a[index];
    point[1] = y_a[index];
    if(dims[2] != 0)
    {
      conduit::float32_array z_a = n_coords["values/z"].value();
      point[2] = z_a[logical_index[2]];
    }

  }

  conduit::Node res;
  res.set(point,3);
  return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

conduit::Node
point_location(const conduit::Node &domain,
               const int &index,
               const std::string topo_name)
{
  std::string topo = topo_name;
  // if we don't specify a topology, find the first topology ...
  if(topo_name == "")
  {
      conduit::NodeConstIterator itr = domain["topologies"].children();
      itr.next();
      topo = itr.name();
  }

  const conduit::Node &n_topo   = domain["topologies"][topo];
  const std::string mesh_type   = n_topo["type"].as_string();
  const std::string coords_name = n_topo["coordset"].as_string();

  const conduit::Node &n_coords = domain["coordsets"][coords_name];

  conduit::Node res;
  if(mesh_type == "uniform")
  {
    res = detail::get_uniform_point(n_coords, index);
  }
  else if(mesh_type == "rectilinear")
  {
    res = detail::get_rectilinear_point(n_coords, index);
  }
  else if(mesh_type == "unstructured" || mesh_type == "structured")
  {
    res = detail::get_explicit_point(n_coords, index);
  }
  else
  {
    ASCENT_ERROR("The Architect: unknown mesh type: '"<<mesh_type<<"'");
  }

  return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
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





