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
#include "ascent_conduit_reductions.hpp"

#include <ascent_logging.hpp>
#include <ascent_mpi_utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

#include <flow_workspace.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#include <mpi.h>
#endif

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
// -- begin ascent::runtime::expressions --
//-----------------------------------------------------------------------------
namespace expressions
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------
namespace detail
{
int
get_num_vertices(const std::string &shape_type)
{
  int num = 0;
  if(shape_type == "tri")
  {
    num = 3;
  }
  else if(shape_type == "quad")
  {
    num = 4;
  }
  else if(shape_type == "tet")
  {
    num = 4;
  }
  else if(shape_type == "hex")
  {
    num = 8;
  }
  else if(shape_type == "point")
  {
    num = 1;
  }
  else
  {
    ASCENT_ERROR("Cannot get the number of vertices for the shape '"
                 << shape_type << "'.");
  }
  return num;
}

template <size_t num_dims>
std::array<size_t, num_dims>
logical_index(const size_t index, const std::array<size_t, num_dims> &dims)
{
  ASCENT_ERROR("Unsupported number of dimensions: " << num_dims);
}

template <>
std::array<size_t, 1>
logical_index(const size_t index, const std::array<size_t, 1> &dims)
{
  return {index};
}

template <>
std::array<size_t, 2>
logical_index(const size_t index, const std::array<size_t, 2> &dims)
{
  return {index % dims[0], index / dims[0]};
}

template <>
std::array<size_t, 3>
logical_index(const size_t index, const std::array<size_t, 3> &dims)
{
  return {index % dims[0],
          (index / dims[0]) % dims[0],
          index / (dims[0] * dims[1])};
}
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- Topology
//-----------------------------------------------------------------------------
Topology::Topology(const std::string &topo_name,
                   const conduit::Node &domain,
                   const size_t num_dims)
    : domain(domain), topo_name(topo_name),
      topo_type(domain["topologies/" + topo_name + "/type"].as_string()),
      coords_name(domain["topologies/" + topo_name + "/coordset"].as_string()),
      coords_type(domain["coordsets/" + coords_name + "/type"].as_string()),
      num_dims(num_dims)
{
}

size_t
Topology::get_num_points() const
{
  return num_points;
}

size_t
Topology::get_num_cells() const
{
  return num_cells;
}
//-----------------------------------------------------------------------------
// -- PointTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
PointTopology<T, N>::PointTopology(const std::string &topo_name,
                                   const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "point")
  {
    ASCENT_ERROR("Cannot initialize a PointTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  if(this->coord_type == "uniform")
  {
    const conduit::Node &n_coords = domain["coordsets/" + this->coords_name];
    const conduit::Node &n_dims = n_coords["dims"];
    const conduit::Node &n_origin = n_coords["origin"];
    const conduit::Node &n_spacing = n_coords["spacing"];
    for(size_t i = 0; i < N; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      const std::string coord = std::string(1, 'x' + i);
      dims[i] = n_dims[dim].to_int();
      origin[i] = n_origin[dim].to_float64();
      spacing[i] = n_spacing["d" + coord].to_float64();
      num_points *= dims[i];
      num_cells *= dims[i] - 1;
    }
  }
  else if(this->coord_type == "rectilinear")
  {
    const conduit::Node &values =
        domain["coordsets/" + this->coords_name + "/values"];
    num_points = 1;
    for(size_t i = 0; i < N; ++i)
    {
      const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
      coords[i] = coord_values.value();
      num_points *= coords[i].dtype().number_of_elements();
    }
  }
  else if(this->coord_type == "explicit")
  {
    const conduit::Node &values =
        domain["coordsets/" + this->coords_name + "/values"];
    for(size_t i = 0; i < N; ++i)
    {
      const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
      coords[i] = coord_values.value();
    }
    num_points = coords[0].dtype().number_of_elements();
  }
  else
  {
    ASCENT_ERROR("Unknown coordinate type '"
                 << this->coord_type << "' for point topology '" << topo_name
                 << "' in domain " << domain.name() << ".");
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
PointTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  if(this->coord_type == "uniform")
  {
    auto l_index = detail::logical_index(index, dims);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = origin[i] + l_index[i] * spacing[i];
    }
  }
  else if(this->coord_type == "rectilinear")
  {
    std::array<size_t, N> dims;
    for(size_t i = 0; i < N; ++i)
    {
      dims[i] = coords[i].number_of_elements();
    }
    const auto l_index = detail::logical_index(index, dims);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = coords[i][l_index[i]];
    }
  }
  else if(this->coord_type == "explicit")
  {
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = coords[i][index];
    }
  }
  else
  {
    ASCENT_ERROR("Unknown coordinate type '"
                 << this->coord_type << "' for point topology '" << topo_name
                 << "' in domain " << domain.name() << ".");
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
PointTopology<T, N>::element_location(const size_t index) const
{
  ASCENT_ERROR("Cannot get the element location of a point topology '"
               << topo_name << "'.");
}

template <typename T, size_t N>
size_t
PointTopology<T, N>::get_num_cells() const
{
  ASCENT_ERROR("Cannot get the number of cells in a point topology '"
               << topo_name << "'.");
}

//-----------------------------------------------------------------------------
// -- UniformTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
UniformTopology<T, N>::UniformTopology(const std::string &topo_name,
                                       const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "uniform")
  {
    ASCENT_ERROR("Cannot initialize a UniformTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  const conduit::Node &n_coords = domain["coordsets/" + this->coords_name];
  const conduit::Node &n_dims = n_coords["dims"];
  const conduit::Node &n_origin = n_coords["origin"];
  const conduit::Node &n_spacing = n_coords["spacing"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const std::string dim = std::string(1, 'i' + i);
    const std::string coord = std::string(1, 'x' + i);
    dims[i] = n_dims[dim].to_int32();
    origin[i] = n_origin[coord].to_float64();
    spacing[i] = n_spacing["d" + coord].to_float64();
    num_points *= dims[i];
    num_cells *= dims[i] - 1;
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UniformTopology<T, N>::vertex_location(const size_t index) const
{
  auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = origin[i] + l_index[i] * spacing[i];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UniformTopology<T, N>::element_location(const size_t index) const
{
  std::array<size_t, N> element_dims;
  for(size_t i = 0; i < N; ++i)
  {
    element_dims[i] = dims[i] - 1;
  }
  const auto l_index = detail::logical_index(index, element_dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = origin[i] + (l_index[i] + 0.5) * spacing[i];
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- RectilinearTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
RectilinearTopology<T, N>::RectilinearTopology(const std::string &topo_name,
                                               const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "rectilinear")
  {
    ASCENT_ERROR("Cannot initialize a RectilinearTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    coords[i] = coord_values.value();
    num_points *= coords[i].dtype().number_of_elements();
    num_cells *= coords[i].dtype().number_of_elements() - 1;
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
RectilinearTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<size_t, N> dims;
  for(size_t i = 0; i < N; ++i)
  {
    dims[i] = coords[i].number_of_elements();
  }
  const auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][l_index[i]];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
RectilinearTopology<T, N>::element_location(const size_t index) const
{
  std::array<size_t, N> dims;
  for(size_t i = 0; i < N; ++i)
  {
    dims[i] = coords[i].number_of_elements() - 1;
  }
  const auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = (coords[i][l_index[i]] + coords[i][l_index[i] + 1]) / 2;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- StructuredTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
StructuredTopology<T, N>::StructuredTopology(const std::string &topo_name,
                                             const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "structured")
  {
    ASCENT_ERROR("Cannot initialize a StructuredTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }
  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  const conduit::Node &n_dims =
      domain["topologies/" + topo_name + "/elements/dims"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    const std::string &dim = std::string(1, 'i' + i);
    coords[i] = coord_values.value();
    // the blueprint gives structured dims in terms of elements not vertices so
    // we change it to vertices so that it's consistent with uniform
    dims[i] = n_dims[dim].to_int32() + 1;
    num_points *= dims[i];
    num_cells *= dims[i] - 1;
  }
  // check that number of vertices in coordset matches dims
  // TODO maybe this is fine and should just be a warning?
  if((size_t)coords[0].dtype().number_of_elements() != num_points)
  {
    ASCENT_ERROR(
        "StructuredTopology ("
        << topo_name << "): The number of points calculated (" << num_points
        << ") differs from the number of vertices in corresponding coordset ("
        << coords[0].dtype().number_of_elements() << ").");
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
StructuredTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][index];
  }
  return loc;
}

constexpr size_t
constexpr_pow(size_t x, size_t y)
{
  return y == 0 ? 1 : x * constexpr_pow(x, y - 1);
}

// vertices are ordered in the VTK format
// https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
template <typename T, size_t N>
std::array<conduit::float64, 3>
StructuredTopology<T, N>::element_location(const size_t index) const
{

  std::array<size_t, N> element_dims;
  for(size_t i = 0; i < N; ++i)
  {
    element_dims[i] = dims[i] - 1;
  }
  const auto element_index = detail::logical_index(index, element_dims);

  constexpr size_t num_vertices = constexpr_pow(2, N);
  std::array<size_t, num_vertices> vertices;
  if(num_vertices == 2)
  {
    vertices[0] = element_index[0];
    vertices[1] = vertices[0] + 1;
  }
  else if(num_vertices == 4)
  {
    vertices[0] = element_index[1] * dims[0] + element_index[0];
    vertices[1] = vertices[0] + 1;
    vertices[2] = vertices[1] + dims[0];
    vertices[3] = vertices[2] - 1;
  }
  else if(num_vertices == 8)
  {
    vertices[0] = (element_index[2] * dims[1] + element_index[1]) * dims[0] +
                  element_index[0];
    vertices[1] = vertices[0] + 1;
    vertices[2] = vertices[1] + dims[0];
    vertices[3] = vertices[2] - 1;
    vertices[4] = vertices[0] + dims[0] * dims[1];
    vertices[5] = vertices[4] + 1;
    vertices[6] = vertices[5] + dims[0];
    vertices[7] = vertices[6] - 1;
  }

  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < num_vertices; ++i)
  {
    const auto vert_loc = vertex_location(vertices[i]);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] += vert_loc[i];
    }
  }
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] /= num_vertices;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- UnstructuredTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
UnstructuredTopology<T, N>::UnstructuredTopology(const std::string &topo_name,
                                                 const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "unstructured")
  {
    ASCENT_ERROR(
        "Cannot initialize a UnstructuredTopology class from topology '"
        << topo_name << "' in domain " << domain.name() << " which has type '"
        << this->topo_type << "'.");
  }
  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    coords[i] = coord_values.value();
  }
  const conduit::Node &elements =
      domain["topologies/" + topo_name + "/elements"];
  shape = elements["shape"].as_string();
  if(shape == "polyhedral")
  {
    polyhedral_connectivity = elements["connectivity"].value();
    polyhedral_sizes = elements["sizes"].value();
    polyhedral_offsets = elements["offsets"].value();
    num_cells = polyhedral_sizes.dtype().number_of_elements();

    const conduit::Node &subelements =
        domain["topologies/" + topo_name + "/subelements"];
    connectivity = subelements["connectivity"].value();
    sizes = subelements["sizes"].value();
    offsets = subelements["offsets"].value();
    polyhedral_shape = subelements["shape"].as_string();
    if(polyhedral_shape != "polygonal")
    {
      polyhedral_shape_size = detail::get_num_vertices(polyhedral_shape);
    }
  }
  else if(shape == "polygonal")
  {
    connectivity = elements["connectivity"].value();
    sizes = elements["sizes"].value();
    offsets = elements["offsets"].value();
    num_cells = sizes.dtype().number_of_elements();
  }
  else
  {
    connectivity = elements["connectivity"].value();
    shape_size = detail::get_num_vertices(shape);
    num_cells = connectivity.dtype().number_of_elements() / shape_size;
  }
}

template <typename T, size_t N>
size_t
UnstructuredTopology<T, N>::get_num_points() const
{
  // number of unique elements in connectivity
  const conduit::int32 *conn_begin = (conduit::int32 *)connectivity.data_ptr();
  const conduit::int32 *conn_end =
      conn_begin + connectivity.dtype().number_of_elements();
  // points used in the topology
  const size_t num_points = std::unordered_set<T>(conn_begin, conn_end).size();
  // points available in the coordset
  const size_t coords_size = domain["coordsets/" + coords_name + "/values"]
                               .child(0)
                               .dtype()
                               .number_of_elements();
  if(num_points != coords_size)
  {
    ASCENT_ERROR("Unstructured topology '"
                 << topo_name << "' has " << coords_size
                 << " points in its associated coordset '" << coords_name
                 << "' but the connectivity "
                    "array only uses "
                 << num_points << " of them.");
  }
  return num_points;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UnstructuredTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][index];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UnstructuredTopology<T, N>::element_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  size_t offset;
  size_t cur_shape_vertices;
  if(shape == "polygonal")
  {
    offset = offsets[index];
    cur_shape_vertices = sizes[index];
  }
  else if(shape == "polyhedral")
  {
    cur_shape_vertices = -1;
    ASCENT_ERROR("element_location for polyhedral shapes is not implemented.");
  }
  else
  {
    offset = index * shape_size;
    cur_shape_vertices = shape_size;
  }
  for(size_t i = 0; i < cur_shape_vertices; ++i)
  {
    const auto vert_loc = vertex_location(connectivity[offset + i]);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] += vert_loc[i];
    }
  }
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] /= cur_shape_vertices;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- topologyFactory
//-----------------------------------------------------------------------------

// make_unique is a c++14 feature
// this is not as general (e.g. doesn't work on array types)
template <typename T, typename... Args>
std::unique_ptr<T>
my_make_unique(Args &&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::unique_ptr<Topology>
topologyFactory(const std::string &topo_name, const conduit::Node &domain)
{
  const conduit::Node &n_topo = domain["topologies/" + topo_name];
  const std::string &topo_type = n_topo["type"].as_string();
  const size_t num_dims = topo_dim(topo_name, domain);
  const std::string type = coord_dtype(topo_name, domain);
  if(topo_type == "uniform")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UniformTopology<conduit::float64, 1>>(topo_name,
                                                                    domain);
        break;
      case 2:
        return my_make_unique<UniformTopology<conduit::float64, 2>>(topo_name,
                                                                    domain);
        break;
      case 3:
        return my_make_unique<UniformTopology<conduit::float64, 3>>(topo_name,
                                                                    domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UniformTopology<conduit::float32, 1>>(topo_name,
                                                                    domain);
        break;
      case 2:
        return my_make_unique<UniformTopology<conduit::float32, 2>>(topo_name,
                                                                    domain);
        break;
      case 3:
        return my_make_unique<UniformTopology<conduit::float32, 3>>(topo_name,
                                                                    domain);
        break;
      }
    }
  }
  else if(topo_type == "rectilinear")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<RectilinearTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<RectilinearTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<RectilinearTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<RectilinearTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<RectilinearTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<RectilinearTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else if(topo_type == "structured")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<StructuredTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<StructuredTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<StructuredTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<StructuredTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<StructuredTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<StructuredTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else if(topo_type == "unstructured")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UnstructuredTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<UnstructuredTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<UnstructuredTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UnstructuredTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<UnstructuredTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<UnstructuredTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else
  {
    ASCENT_ERROR("The Architect: Unsupported topology type '" << topo_type
                                                              << "'.");
  }
  ASCENT_ERROR("topologyFactory returning nullptr, this should never happen.");
  return nullptr;
}
//-----------------------------------------------------------------------------

std::string
possible_components(const conduit::Node &dataset, const std::string &field_name)
{
  std::string res;
  if(dataset.number_of_children() > 0)
  {
    const conduit::Node &dom = dataset.child(0);
    if(dom.has_path("fields/" + field_name))
    {
      if(dom["fields/" + field_name + "/values"].number_of_children() > 0)
      {
        std::vector<std::string> names =
            dom["fields/" + field_name + "/values"].child_names();
        std::stringstream ss;
        ss << "[";
        bool first = true;
        for(auto name : names)
        {
          if(!first)
          {
            ss << ", ";
          }
          ss << name;
        }
        ss << "]";
        res = ss.str();
      }
    }
  }
  return res;
}

bool
is_scalar_field(const conduit::Node &dataset, const std::string &field_name)
{
  bool is_scalar = false;
  bool has_field = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_field && dom.has_path("fields/" + field_name))
    {
      has_field = true;
      const conduit::Node &n_field = dom["fields/" + field_name];
      const int num_children = n_field["values"].number_of_children();
      if(num_children == 0)
      {
        is_scalar = true;
      }
    }
  }
  // check to see if the scalar field exists in any rank
  is_scalar = global_someone_agrees(is_scalar);
  return is_scalar;
}

bool
has_field(const conduit::Node &dataset, const std::string &field_name)
{
  bool has_field = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_field && dom.has_path("fields/" + field_name))
    {
      has_field = true;
    }
  }
  // check to see if the field exists in any rank
  has_field = global_someone_agrees(has_field);
  return has_field;
}

bool
has_topology(const conduit::Node &dataset, const std::string &topo_name)
{
  bool has_topo = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_topo && dom.has_path("topologies/" + topo_name))
    {
      has_topo = true;
    }
  }
  // check to see if the topology exists in any rank
  has_topo = global_someone_agrees(has_topo);
  return has_topo;
}

std::string
known_topos(const conduit::Node &dataset)
{
  std::vector<std::string> names = dataset.child(0)["topologies"].child_names();
  std::stringstream ss;
  ss << "[";
  for(size_t i = 0; i < names.size(); ++i)
  {
    ss << names[i];
    if(i < names.size() - 1)
    {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string
known_fields(const conduit::Node &dataset)
{
  std::vector<std::string> names = dataset.child(0)["fields"].child_names();
  std::stringstream ss;
  ss << "[";
  for(size_t i = 0; i < names.size(); ++i)
  {
    ss << names[i];
    if(i < names.size() - 1)
    {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

bool
has_component(const conduit::Node &dataset,
              const std::string &field_name,
              const std::string &component)
{
  bool has_comp = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_comp &&
       dom.has_path("fields/" + field_name + "/values/" + component))
    {
      has_comp = true;
    }
  }
  // check to see if the field exists in any rank
  has_comp = global_someone_agrees(has_comp);
  return has_comp;
}

// TODO If someone names their fields x,y,z things will go wrong
bool
is_xyz(const std::string &axis_name)
{
  return axis_name == "x" || axis_name == "y" || axis_name == "z";
}

// returns -1 if value lies outside the range
int
get_bin_index(const conduit::float64 value, const conduit::Node &axis)
{
  const bool clamp = axis["clamp"].to_uint8();
  if(axis.has_path("bins"))
  {
    // rectilinear
    const conduit::float64 *bins_begin = axis["bins"].value();
    const conduit::float64 *bins_end =
        bins_begin + axis["bins"].dtype().number_of_elements() - 1;
    // first element greater than value
    const conduit::float64 *res = std::upper_bound(bins_begin, bins_end, value);
    if(clamp)
    {
      if(res <= bins_begin)
      {
        return 0;
      }
      else if(res >= bins_end)
      {
        return bins_end - bins_begin - 1;
      }
    }
    else if(res <= bins_begin || res >= bins_end)
    {
      return -1;
    }
    return (res - 1) - bins_begin;
  }
  // uniform
  const double inv_delta =
      axis["num_bins"].to_float64() /
      (axis["max_val"].to_float64() - axis["min_val"].to_float64());
  const int bin_index =
      static_cast<int>((value - axis["min_val"].to_float64()) * inv_delta);
  if(clamp)
  {
    if(bin_index < 0)
    {
      return 0;
    }
    else if(bin_index >= axis["num_bins"].as_int32())
    {
      return axis["num_bins"].as_int32() - 1;
    }
  }
  else if(bin_index < 0 || bin_index >= axis["num_bins"].as_int32())
  {
    return -1;
  }
  return bin_index;
}

// returns -1 if value lies outside the range
template <typename T>
int
uniform_bin(const T value,
            const T min_val,
            const T max_val,
            const size_t num_bins,
            const bool clamp)
{
  const double inv_delta = num_bins / (max_val - min_val);
  const size_t bin_index = static_cast<size_t>((value - min_val) * inv_delta);
  if(clamp)
  {
    if(bin_index < 0)
    {
      return 0;
    }
    else if(bin_index >= num_bins)
    {
      return num_bins - 1;
    }
  }
  else if(bin_index < 0 || bin_index >= num_bins)
  {
    return -1;
  }
  return bin_index;
}

template <typename T>
int
rectilinear_bin(const T value,
                const T *const bins_begin,
                const T *const bins_end,
                const bool clamp)
{
  const T *const res = std::upper_bound(bins_begin, bins_end, value);
  if(clamp)
  {
    if(res <= bins_begin)
    {
      return 0;
    }
    else if(res >= bins_end)
    {
      return bins_end - bins_begin - 2;
    }
  }
  else if(res <= bins_begin || res >= bins_end)
  {
    return -1;
  }
  return (res - 1) - bins_begin;
}

template <typename T>
void
field_homes(const T *const field,
            int *const homes,
            const size_t num_homes,
            const size_t stride,
            const conduit::Node &axis)
{
  const bool clamp = axis["clamp"].to_uint8();
  if(axis.has_path("bins"))
  {
    // rectilinear
    const T *bins_begin = axis["bins"].value();
    const T *bins_end = bins_begin + axis["bins"].dtype().number_of_elements();
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_homes; ++i)
    {
      const int bin_index =
          rectilinear_bin(field[i], bins_begin, bins_end, clamp);
      if(bin_index != -1)
      {
        homes[i] += bin_index * stride;
      }
      else
      {
        homes[i] = -1;
      }
    }
  }
  else
  {
    const T min_val = axis["min_val"].to_float64();
    const T max_val = axis["max_val"].to_float64();
    const T num_bins = axis["num_bins"].to_float64();
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_homes; ++i)
    {
      const int bin_index =
          uniform_bin(field[i], min_val, max_val, num_bins, clamp);
      if(bin_index != -1)
      {
        homes[i] += bin_index * stride;
      }
      else
      {
        homes[i] = -1;
      }
    }
  }
}

void
populate_homes(const conduit::Node &dom,
               const conduit::Node &bin_axes,
               const std::string &topo_name,
               const std::string &assoc_str,
               conduit::Node &res)
{
  int num_axes = bin_axes.number_of_children();

  // ensure this domain has the necessary fields
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    const conduit::Node &axis = bin_axes.child(axis_index);
    const std::string axis_name = axis.name();
    if(!dom.has_path("fields/" + axis_name) && !is_xyz(axis_name))
    {
      // return an error and skip the domain in binning
      conduit::Node res;
      res["error/field_name"] = axis_name;
      return;
    }
  }

  std::unique_ptr<Topology> topo = topologyFactory(topo_name, dom);

  // Calculate the size of homes
  size_t homes_size = 0;
  if(assoc_str == "vertex")
  {
    homes_size = topo->get_num_points();
  }
  else if(assoc_str == "element")
  {
    homes_size = topo->get_num_cells();
  }

  // if we need to bin a spacial axis get coordinates for each element
  bool has_spacial = false;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    const conduit::Node &axis = bin_axes.child(axis_index);
    const std::string axis_name = axis.name();
    if(is_xyz(axis_name))
    {
      has_spacial = true;
      break;
    }
  }

  std::array<conduit::float64 *, 3> coords;
  if(has_spacial)
  {
    coords[0] = new conduit::float64[homes_size];
    coords[1] = new conduit::float64[homes_size];
    coords[2] = new conduit::float64[homes_size];
    std::array<conduit::float64, 3> loc;
    if(assoc_str == "vertex")
    {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < homes_size; ++i)
      {
        loc = topo->vertex_location(i);
        coords[0][i] = loc[0];
        coords[1][i] = loc[1];
        coords[2][i] = loc[2];
      }
    }
    else if(assoc_str == "element")
    {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < homes_size; ++i)
      {
        loc = topo->element_location(i);
        coords[0][i] = loc[0];
        coords[1][i] = loc[1];
        coords[2][i] = loc[2];
      }
    }
  }

  // each domain has a homes array
  // homes maps each datapoint (or cell) to an index in bins
  res.set(conduit::DataType::c_int(homes_size));
  int *homes = res.value();

  int stride = 1;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    const conduit::Node &axis = bin_axes.child(axis_index);
    const std::string axis_name = axis.name();
    if(dom.has_path("fields/" + axis_name))
    {
      std::string values_path = "fields/" + axis_name + "/values";
      if(dom[values_path].dtype().is_float32())
      {
        const conduit::float32 *values = dom[values_path].value();
        field_homes(values, homes, homes_size, stride, axis);
      }
      else
      {
        const conduit::float64 *values = dom[values_path].value();
        field_homes(values, homes, homes_size, stride, axis);
      }
    }
    else if(is_xyz(axis_name))
    {
      int coord = axis_name[0] - 'x';
      field_homes(coords[coord], homes, homes_size, stride, axis);
    }

    if(bin_axes.child(axis_index).has_path("num_bins"))
    {
      // uniform axis
      stride *= bin_axes.child(axis_index)["num_bins"].as_int32();
    }
    else
    {
      // rectilinear axis
      stride *=
          bin_axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;
    }
  }
  if(has_spacial)
  {
    delete[] coords[0];
    delete[] coords[1];
    delete[] coords[2];
  }
}

void
update_bin(double *bins,
           const int i,
           const int num_bins,
           const double value,
           const std::string &reduction_op)
{
  if(reduction_op == "min")
  {
    // have to keep track of count anyways in order to detect which bins are
    // empty
    bins[i] = std::min(bins[i], value);
    bins[num_bins + i] += 1;
  }
  else if(reduction_op == "max")
  {
    bins[i] = std::max(bins[i], value);
    bins[num_bins + i] += 1;
  }
  else if(reduction_op == "avg" || reduction_op == "sum" ||
          reduction_op == "pdf")
  {
    bins[i] += value;
    bins[num_bins + i] += 1;
  }
  else if(reduction_op == "rms")
  {
    bins[i] += value * value;
    bins[num_bins + i] += 1;
  }
  else if(reduction_op == "var" || reduction_op == "std")
  {
    bins[i] += value * value;
    bins[num_bins + i] += value;
    bins[2 * num_bins + i] += 1;
  }
}

// get the association and topology and ensure they are the same
conduit::Node
global_topo_and_assoc(const conduit::Node &dataset,
                      const std::vector<std::string> var_names)
{
  std::string assoc_str;
  std::string topo_name;
  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    for(const std::string &var_name : var_names)
    {
      if(dom.has_path("fields/" + var_name))
      {
        const std::string cur_assoc_str =
            dom["fields/" + var_name + "/association"].as_string();
        if(assoc_str.empty())
        {
          assoc_str = cur_assoc_str;
        }
        else if(assoc_str != cur_assoc_str)
        {
          ASCENT_ERROR("All Binning fields must have the same association.");
        }

        const std::string cur_topo_name =
            dom["fields/" + var_name + "/topology"].as_string();
        if(topo_name.empty())
        {
          topo_name = cur_topo_name;
        }
        else if(topo_name != cur_topo_name)
        {
          ASCENT_ERROR("All Binning fields must have the same topology.");
        }
      }
    }
  }
#ifdef ASCENT_MPI_ENABLED
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);

  struct MaxLoc
  {
    double size;
    int rank;
  };

  // there is no MPI_INT_INT so shove the "small" size into double
  MaxLoc maxloc = {(double)topo_name.length(), rank};
  MaxLoc maxloc_res;
  MPI_Allreduce(&maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);

  conduit::Node msg;
  msg["assoc_str"] = assoc_str;
  msg["topo_name"] = topo_name;
  conduit::relay::mpi::broadcast_using_schema(msg, maxloc_res.rank, mpi_comm);

  if(assoc_str != msg["assoc_str"].as_string())
  {
    ASCENT_ERROR("All Binning fields must have the same association.");
  }
  if(topo_name != msg["topo_name"].as_string())
  {
    ASCENT_ERROR("All Binning fields must have the same topology.");
  }
#endif
  if(assoc_str.empty())
  {
    ASCENT_ERROR("Could not determine the associate from the given "
                 "reduction_var and axes. Try supplying a field.");
  }
  else if(assoc_str != "vertex" && assoc_str != "element")
  {
    ASCENT_ERROR("Unknown association: '"
                 << assoc_str
                 << "'. Binning only supports vertex and element association.");
  }

  conduit::Node res;
  res["topo_name"] = topo_name;
  res["assoc_str"] = assoc_str;
  return res;
}

// returns a Node containing the min, max and dim for x,y,z given a topology
conduit::Node
global_bounds(const conduit::Node &dataset, const std::string &topo_name)
{
  double min_coords[3] = {std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max()};
  double max_coords[3] = {std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest()};
  const std::string axes[3][3] = {
      {"x", "i", "dx"}, {"y", "j", "dy"}, {"z", "k", "dz"}};
  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    const conduit::Node &n_topo = dom["topologies/" + topo_name];
    const std::string topo_type = n_topo["type"].as_string();
    const std::string coords_name = n_topo["coordset"].as_string();
    const conduit::Node &n_coords = dom["coordsets/" + coords_name];

    if(topo_type == "uniform")
    {
      int num_dims = n_coords["dims"].number_of_children();
      for(int i = 0; i < num_dims; ++i)
      {
        double origin = n_coords["origin/" + axes[i][0]].to_float64();
        int dim = n_coords["dims/" + axes[i][1]].to_int();
        double spacing = n_coords["spacing/" + axes[i][2]].to_float64();

        min_coords[i] = std::min(min_coords[i], origin);
        max_coords[i] = std::max(min_coords[i], origin + (dim - 1) * spacing);
      }
    }
    else if(topo_type == "rectilinear" || topo_type == "structured" ||
            topo_type == "unstructured")
    {
      int num_dims = n_coords["values"].number_of_children();
      for(int i = 0; i < num_dims; ++i)
      {
        const std::string axis_path = "values/" + axes[i][0];
        min_coords[i] = std::min(
            min_coords[i], array_min(n_coords[axis_path])["value"].as_double());
        max_coords[i] = std::max(
            max_coords[i], array_max(n_coords[axis_path])["value"].as_double());
      }
    }
    else
    {
      ASCENT_ERROR("The Architect: unknown topology type: '" << topo_type
                                                             << "'");
    }
  }
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Allreduce(MPI_IN_PLACE, min_coords, 3, MPI_DOUBLE, MPI_MIN, mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, max_coords, 3, MPI_DOUBLE, MPI_MAX, mpi_comm);
#endif
  conduit::Node res;
  res["max_coords"].set(max_coords, 3);
  res["min_coords"].set(min_coords, 3);
  return res;
}

void
init_bins(double *bins, const int size, const std::string reduction_op)
{
  if(reduction_op != "max" && reduction_op != "min")
  {
    // already init to 0, so do nothing
    return;
  }

  double init_val;
  if(reduction_op == "max")
  {
    init_val = std::numeric_limits<double>::lowest();
  }
  else
  {
    init_val = std::numeric_limits<double>::max();
  }

#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    bins[i] = init_val;
  }
}

// reduction_op: sum, min, max, avg, pdf, std, var, rms
conduit::Node
binning(const conduit::Node &dataset,
        conduit::Node &bin_axes,
        const std::string &reduction_var,
        const std::string &reduction_op,
        const double empty_bin_val,
        const std::string &component)
{
  std::vector<std::string> var_names = bin_axes.child_names();
  if(!reduction_var.empty())
  {
    var_names.push_back(reduction_var);
  }
  const conduit::Node &topo_and_assoc =
      global_topo_and_assoc(dataset, var_names);
  const std::string topo_name = topo_and_assoc["topo_name"].as_string();
  const std::string assoc_str = topo_and_assoc["assoc_str"].as_string();

  const conduit::Node &bounds = global_bounds(dataset, topo_name);
  const double *min_coords = bounds["min_coords"].value();
  const double *max_coords = bounds["max_coords"].value();
  const std::string axes[3][3] = {
      {"x", "i", "dx"}, {"y", "j", "dy"}, {"z", "k", "dz"}};
  // populate min_val, max_val, for x,y,z
  for(int axis_num = 0; axis_num < 3; ++axis_num)
  {
    if(bin_axes.has_path(axes[axis_num][0]))
    {
      conduit::Node &axis = bin_axes[axes[axis_num][0]];

      if(axis.has_path("bins"))
      {
        // rectilinear binning was specified
        continue;
      }

      if(min_coords[axis_num] == std::numeric_limits<double>::max())
      {
        ASCENT_ERROR("Could not finds bounds for axis: "
                     << axes[axis_num][0]
                     << ". It probably doesn't exist in the topology: "
                     << topo_name);
      }

      if(!axis.has_path("min_val"))
      {
        axis["min_val"] = min_coords[axis_num];
      }

      if(!axis.has_path("max_val"))
      {
        // We add eps because the last bin isn't inclusive
        double min_val = axis["min_val"].to_float64();
        double length = max_coords[axis_num] - min_val;
        double eps = length * 1e-8;
        axis["max_val"] = max_coords[axis_num] + eps;
      }
    }
  }

  int num_axes = bin_axes.number_of_children();

  // create bins
  size_t num_bins = 1;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    if(bin_axes.child(axis_index).has_path("num_bins"))
    {
      // uniform axis
      num_bins *= bin_axes.child(axis_index)["num_bins"].as_int32();
    }
    else
    {
      // rectilinear axis
      num_bins *=
          bin_axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;
    }
  }
  // number of variables held per bin (e.g. sum and cnt for average)
  int num_bin_vars = 2;
  if(reduction_op == "var" || reduction_op == "std")
  {
    num_bin_vars = 3;
  }
  const int bins_size = num_bins * num_bin_vars;
  double *bins = new double[bins_size]();
  init_bins(bins, num_bins, reduction_op);

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);

    conduit::Node n_homes;
    populate_homes(dom, bin_axes, topo_name, assoc_str, n_homes);

    if(n_homes.has_path("error"))
    {
      ASCENT_INFO("Binning: not binning domain "
                  << dom_index << " because field: '"
                  << n_homes["error/field_name"].to_string()
                  << "' was not found.");
      continue;
    }
    const int *homes = n_homes.as_int_ptr();
    const int homes_size = n_homes.dtype().number_of_elements();

    // update bins
    if(reduction_var.empty())
    {
      //#ifdef ASCENT_USE_OPENMP
      //#pragma omp parallel for
      //#endif
      for(int i = 0; i < homes_size; ++i)
      {
        if(homes[i] != -1)
        {
          update_bin(bins, homes[i], num_bins, 1, reduction_op);
        }
      }
    }
    else if(dom.has_path("fields/" + reduction_var))
    {
      const std::string comp_path = component == "" ? "" : "/" + component;
      const std::string values_path =
          "fields/" + reduction_var + "/values" + comp_path;

      if(dom[values_path].dtype().is_float32())
      {
        const conduit::float32_array values = dom[values_path].value();
        //#ifdef ASCENT_USE_OPENMP
        //#pragma omp parallel for
        //#endif
        for(int i = 0; i < homes_size; ++i)
        {
          if(homes[i] != -1)
          {
            update_bin(bins, homes[i], num_bins, values[i], reduction_op);
          }
        }
      }
      else
      {
        const conduit::float64_array values = dom[values_path].value();
        //#ifdef ASCENT_USE_OPENMP
        //#pragma omp parallel for
        //#endif
        for(int i = 0; i < homes_size; ++i)
        {
          if(homes[i] != -1)
          {
            update_bin(bins, homes[i], num_bins, values[i], reduction_op);
          }
        }
      }
    }
    else if(is_xyz(reduction_var))
    {
      int coord = reduction_var[0] - 'x';
      std::unique_ptr<Topology> t = topologyFactory(topo_name, dom);
      std::array<conduit::float64, 3> loc;
      //#ifdef ASCENT_USE_OPENMP
      //#pragma omp parallel for
      //#endif
      for(int i = 0; i < homes_size; ++i)
      {
        if(assoc_str == "vertex")
        {
          loc = t->vertex_location(i);
        }
        else if(assoc_str == "element")
        {
          loc = t->element_location(i);
        }
        if(homes[i] != -1)
        {
          update_bin(bins, homes[i], num_bins, loc[coord], reduction_op);
        }
      }
    }
    else
    {
      ASCENT_INFO("Binning: not binning domain "
                  << dom_index << " because field: '" << reduction_var
                  << "' was not found.");
    }
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  double *global_bins = new double[bins_size];
  if(reduction_op == "sum" || reduction_op == "pdf" || reduction_op == "avg" ||
     reduction_op == "std" || reduction_op == "var" || reduction_op == "rms")
  {
    MPI_Allreduce(bins, global_bins, bins_size, MPI_DOUBLE, MPI_SUM, mpi_comm);
  }
  else if(reduction_op == "min")
  {
    MPI_Allreduce(bins, global_bins, num_bins, MPI_DOUBLE, MPI_MIN, mpi_comm);
    MPI_Allreduce(bins + num_bins,
                  global_bins + num_bins,
                  num_bins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_comm);
  }
  else if(reduction_op == "max")
  {
    MPI_Allreduce(bins, global_bins, num_bins, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(bins + num_bins,
                  global_bins + num_bins,
                  num_bins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_comm);
  }
  delete[] bins;
  bins = global_bins;
#endif

  conduit::Node res;
  res["value"].set(conduit::DataType::c_double(num_bins));
  double *res_bins = res["value"].value();
  if(reduction_op == "pdf")
  {
    double total = 0;
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for reduction(+ : total)
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      total += bins[i];
    }
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      if(bins[num_bins + i] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i] / total;
      }
    }
  }
  else if(reduction_op == "sum" || reduction_op == "min" ||
          reduction_op == "max")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      if(bins[num_bins + i] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i];
      }
    }
  }
  else if(reduction_op == "avg")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[i];
      const double n = bins[num_bins + i];
      if(n == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = sumX / n;
      }
    }
  }
  else if(reduction_op == "rms")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[i];
      const double n = bins[num_bins + i];
      if(n == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = std::sqrt(sumX / n);
      }
    }
  }
  else if(reduction_op == "var")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[i];
      const double sumX = bins[num_bins + i];
      const double n = bins[2 * num_bins + i];
      if(n == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = (sumX2 / n) - std::pow(sumX / n, 2);
      }
    }
  }
  else if(reduction_op == "std")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[i];
      const double sumX = bins[num_bins + i];
      const double n = bins[2 * num_bins + i];
      if(n == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = std::sqrt((sumX2 / n) - std::pow(sumX / n, 2));
      }
    }
  }
  res["association"] = assoc_str;
  delete[] bins;
  return res;
}

conduit::Node
field_histogram(const conduit::Node &dataset,
                const std::string &field,
                const double &min_val,
                const double &max_val,
                const int &num_bins)
{

  double *bins = new double[num_bins]();

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_histogram(dom[path], min_val, max_val, num_bins);

      double *dom_hist = res["value"].value();
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
      for(int bin_index = 0; bin_index < num_bins; ++bin_index)
      {
        bins[bin_index] += dom_hist[bin_index];
      }
    }
  }
  conduit::Node res;

#ifdef ASCENT_MPI_ENABLED
  double *global_bins = new double[num_bins];

  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Allreduce(bins, global_bins, num_bins, MPI_INT, MPI_SUM, mpi_comm);

  delete[] bins;
  bins = global_bins;
#endif
  res["value"].set(bins, num_bins);
  res["min_val"] = min_val;
  res["max_val"] = max_val;
  res["num_bins"] = num_bins;
  delete[] bins;
  return res;
}

void
paint_binning(const conduit::Node &binning,
              conduit::Node &dataset,
              const std::string &field_name,
              const std::string &topo_name,
              const std::string &assoc_str,
              const double default_value)
{
  const conduit::Node &bin_axes = binning["attrs/bin_axes/value"];

  // get assoc_str and topo_name
  std::vector<std::string> axis_names = bin_axes.child_names();
  bool all_xyz = true;
  for(const std::string &axis_name : axis_names)
  {
    all_xyz &= is_xyz(axis_name);
  }

  std::string new_topo_name;
  std::string new_assoc_str;
  if(all_xyz)
  {
    if(!topo_name.empty())
    {
      new_topo_name = topo_name;
    }
    else if(dataset.child(0)["topologies"].number_of_children() == 1)
    {
      new_topo_name = dataset.child(0)["topologies"].child(0).name();
    }
    else
    {
      ASCENT_ERROR(
          "Please specify a topology to paint onto. The topology could not "
          "be inferred because the bin axes are a subset of x, y, z. Known "
          "topologies: "
          << known_topos(dataset));
    }

    if(!assoc_str.empty())
    {
      new_assoc_str = assoc_str;
    }
    else
    {
      // and use the association from the binning
      new_assoc_str = binning["attrs/association/value"].as_string();
    }
  }
  else
  {
    const conduit::Node &topo_and_assoc =
        global_topo_and_assoc(dataset, axis_names);
    new_topo_name = topo_and_assoc["topo_name"].as_string();
    new_assoc_str = topo_and_assoc["assoc_str"].as_string();
    if(!topo_name.empty() && topo_name != new_topo_name)
    {
      ASCENT_ERROR(
          "The specified topology '"
          << topo_name
          << "' does not have the required fields specified in the bin axes: "
          << bin_axes.to_yaml() << "\n Did you mean to use '" << new_topo_name
          << "'?");
    }
    if(!assoc_str.empty() && assoc_str != new_assoc_str)
    {
      ASCENT_ERROR(
          "The specified association '"
          << assoc_str
          << "' conflicts with the association of the fields of the bin axes:"
          << bin_axes.to_yaml() << ". Did you mean to use '" << new_assoc_str
          << "'?");
    }
  }

  const double *bins = binning["attrs/value/value"].as_double_ptr();

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    conduit::Node &dom = dataset.child(dom_index);

    conduit::Node n_homes;
    populate_homes(dom, bin_axes, new_topo_name, new_assoc_str, n_homes);
    if(n_homes.has_path("error"))
    {
      ASCENT_INFO("Binning: not painting domain "
                  << dom_index << " because field: '"
                  << n_homes["error/field_name"].to_string()
                  << "' was not found.");
      continue;
    }
    const int *homes = n_homes.as_int_ptr();
    const int homes_size = n_homes.dtype().number_of_elements();

    dom["fields/" + field_name + "/association"] = new_assoc_str;
    dom["fields/" + field_name + "/topology"] = new_topo_name;
    dom["fields/" + field_name + "/values"].set(
        conduit::DataType::float64(homes_size));
    conduit::float64_array values =
        dom["fields/" + field_name + "/values"].value();
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < homes_size; ++i)
    {
      if(homes[i] != -1)
      {
        values[i] = bins[homes[i]];
      }
      else
      {
        values[i] = default_value;
      }
    }
  }

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh", dataset, info))
  {
    dataset.print();
    info.print();
    ASCENT_ERROR(
        "Failed to verify mesh after painting binning back on the mesh.");
  }
}

void
binning_mesh(const conduit::Node &binning,
             conduit::Node &mesh,
             const std::string &field_name)
{
  int num_axes = binning["attrs/bin_axes/value"].number_of_children();

  if(num_axes > 3)
  {
    ASCENT_ERROR(
        "Binning mesh: can only construct meshes with 3 or fewer axes.");
  }

  const std::string axes[3][3] = {
      {"x", "i", "dx"}, {"y", "j", "dy"}, {"z", "k", "dz"}};
  // create coordinate set turn uniform axes to rectiliear
  const std::string coords_name = field_name + "_coords";
  mesh["coordsets/" + coords_name + "/type"] = "rectilinear";
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = binning["attrs/bin_axes/value"].child(i);
    if(axis.has_path("bins"))
    {
      // rectilinear
      mesh["coordsets/" + coords_name + "/values/" + axes[i][0]] = axis["bins"];
    }
    else
    {
      // uniform
      const int dim = axis["num_bins"].as_int32() + 1;
      const double delta =
          (axis["max_val"].to_float64() - axis["min_val"].to_float64()) /
          (dim - 1);
      mesh["coordsets/" + coords_name + "/values/" + axes[i][0]].set(
          conduit::DataType::c_double(dim));
      double *bins =
          mesh["coordsets/" + coords_name + "/values/" + axes[i][0]].value();
      for(int j = 0; j < dim; ++j)
      {
        bins[j] = axis["min_val"].to_float64() + j * delta;
      }
    }
  }

  // create topology
  const std::string topo_name = field_name + "_topo";
  mesh["topologies/" + topo_name + "/type"] = "rectilinear";
  mesh["topologies/" + topo_name + "/coordset"] = coords_name;

  // create field
  mesh["fields/" + field_name + "/association"] = "element";
  mesh["fields/" + field_name + "/topology"] = topo_name;
  mesh["fields/" + field_name + "/values"].set(binning["attrs/value/value"]);

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh", mesh, info))
  {
    mesh.print();
    info.print();
    ASCENT_ERROR("Failed to create valid binning mesh.");
  }
}

conduit::Node
field_entropy(const conduit::Node &hist)
{
  const double *hist_bins = hist["attrs/value/value"].value();
  const int num_bins = hist["attrs/num_bins/value"].to_int32();
  double sum = array_sum(hist["attrs/value/value"])["value"].to_float64();
  double entropy = 0;

#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for reduction(+ : entropy)
#endif
  for(int b = 0; b < num_bins; ++b)
  {
    if(hist_bins[b] != 0)
    {
      double p = hist_bins[b] / sum;
      entropy += -p * std::log(p);
    }
  }

  conduit::Node res;
  res["value"] = entropy;
  return res;
}

conduit::Node
field_pdf(const conduit::Node &hist)
{
  const double *hist_bins = hist["attrs/value/value"].value();
  const int num_bins = hist["attrs/num_bins/value"].to_int32();
  double min_val = hist["attrs/min_val/value"].to_float64();
  double max_val = hist["attrs/max_val/value"].to_float64();

  double sum = array_sum(hist["attrs/value/value"])["value"].to_float64();

  double *pdf = new double[num_bins]();

#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
  for(int b = 0; b < num_bins; ++b)
  {
    pdf[b] = hist_bins[b] / sum;
  }

  conduit::Node res;
  res["value"].set(pdf, num_bins);
  res["min_val"] = min_val;
  res["max_val"] = max_val;
  res["num_bins"] = num_bins;
  delete[] pdf;
  return res;
}

conduit::Node
field_cdf(const conduit::Node &hist)
{
  const double *hist_bins = hist["attrs/value/value"].value();
  const int num_bins = hist["attrs/num_bins/value"].to_int32();
  double min_val = hist["attrs/min_val/value"].to_float64();
  double max_val = hist["attrs/max_val/value"].to_float64();

  double sum = array_sum(hist["attrs/value/value"])["value"].to_float64();

  double rolling_cdf = 0;

  double *cdf = new double[num_bins]();

  // TODO can this be parallel?
  for(int b = 0; b < num_bins; ++b)
  {
    rolling_cdf += hist_bins[b] / sum;
    cdf[b] = rolling_cdf;
  }

  conduit::Node res;
  res["value"].set(cdf, num_bins);
  res["min_val"] = min_val;
  res["max_val"] = max_val;
  res["num_bins"] = num_bins;
  delete[] cdf;
  return res;
}

// this only makes sense on a count histogram
conduit::Node
quantile(const conduit::Node &cdf,
         const double val,
         const std::string &interpolation)
{
  const double *cdf_bins = cdf["attrs/value/value"].value();
  const int num_bins = cdf["attrs/num_bins/value"].to_int32();
  double min_val = cdf["attrs/min_val/value"].to_float64();
  double max_val = cdf["attrs/max_val/value"].to_float64();

  conduit::Node res;

  int bin = 0;

  for(; cdf_bins[bin] < val; ++bin)
    ;
  // we overshot
  if(cdf_bins[bin] > val)
    --bin;
  // i and j are the bin boundaries
  double i = min_val + bin * (max_val - min_val) / num_bins;
  double j = min_val + (bin + 1) * (max_val - min_val) / num_bins;

  if(interpolation == "linear")
  {
    if(cdf_bins[bin + 1] - cdf_bins[bin] == 0)
    {
      res["value"] = i;
    }
    else
    {
      res["value"] = i + (j - i) * (val - cdf_bins[bin]) /
                             (cdf_bins[bin + 1] - cdf_bins[bin]);
    }
  }
  else if(interpolation == "lower")
  {
    res["value"] = i;
  }
  else if(interpolation == "higher")
  {
    res["value"] = j;
  }
  else if(interpolation == "midpoint")
  {
    res["value"] = (i + j) / 2;
  }
  else if(interpolation == "nearest")
  {
    res["value"] = (val - i < j - val) ? i : j;
  }

  return res;
}

conduit::Node
field_nan_count(const conduit::Node &dataset, const std::string &field)
{
  double nan_count = 0;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_nan_count(dom[path]);
      nan_count += res["value"].to_float64();
    }
  }
  conduit::Node res;
  res["value"] = nan_count;

  return res;
}

conduit::Node
field_inf_count(const conduit::Node &dataset, const std::string &field)
{
  double inf_count = 0;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_inf_count(dom[path]);
      inf_count += res["value"].to_float64();
    }
  }
  conduit::Node res;
  res["value"] = inf_count;

  return res;
}

conduit::Node
field_min(const conduit::Node &dataset, const std::string &field)
{
  double min_value = std::numeric_limits<double>::max();

  std::string topo_name;
  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_min(dom[path]);
      double a_min = res["value"].to_float64();
      if(a_min < min_value)
      {
        min_value = a_min;
        index = res["index"].as_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
        topo_name = dom["fields/" + field + "/topology"].as_string();
      }
    }
  }

  const std::string assoc_str =
      dataset.child(0)["fields/" + field + "/association"].as_string();

  std::unique_ptr<Topology> t =
      topologyFactory(topo_name, dataset.child(domain));
  std::array<conduit::float64, 3> aloc;
  if(assoc_str == "vertex")
  {
    aloc = t->vertex_location(index);
  }
  else if(assoc_str == "element")
  {
    aloc = t->element_location(index);
  }
  else
  {
    ASCENT_ERROR("Location for " << assoc_str << " not implemented");
  }
  conduit::Node loc;
  loc.set(aloc.data(), 3);

  int rank = 0;
  conduit::Node res;
#ifdef ASCENT_MPI_ENABLED
  struct MinLoc
  {
    double value;
    int rank;
  };

  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);

  MinLoc minloc = {min_value, rank};
  MinLoc minloc_res;
  MPI_Allreduce(&minloc, &minloc_res, 1, MPI_DOUBLE_INT, MPI_MINLOC, mpi_comm);
  min_value = minloc.value;

  double *ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, minloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, minloc_res.rank, mpi_comm);
  MPI_Bcast(&index, 1, MPI_INT, minloc_res.rank, mpi_comm);

  loc.set(ploc, 3);

  rank = minloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["index"] = index;
  res["assoc"] = assoc_str;
  res["position"] = loc;
  res["value"] = min_value;

  return res;
}

conduit::Node
field_sum(const conduit::Node &dataset, const std::string &field)
{

  double sum = 0.;
  long long int count = 0;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_sum(dom[path]);

      double a_sum = res["value"].to_float64();
      long long int a_count = res["count"].to_int64();

      sum += a_sum;
      count += a_count;
    }
  }

#ifdef ASCENT_MPI_ENABLED
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);
  double global_sum;
  MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

  long long int global_count;
  MPI_Allreduce(&count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, mpi_comm);

  sum = global_sum;
  count = global_count;
#endif

  conduit::Node res;
  res["value"] = sum;
  res["count"] = count;
  return res;
}

conduit::Node
field_avg(const conduit::Node &dataset, const std::string &field)
{
  conduit::Node sum = field_sum(dataset, field);

  double avg = sum["value"].to_float64() / sum["count"].to_float64();

  conduit::Node res;
  res["value"] = avg;
  return res;
}

conduit::Node
field_max(const conduit::Node &dataset, const std::string &field)
{
  double max_value = std::numeric_limits<double>::lowest();

  std::string topo_name;
  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_max(dom[path]);
      double a_max = res["value"].to_float64();
      if(a_max > max_value)
      {
        max_value = a_max;
        index = res["index"].as_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
        topo_name = dom["fields/" + field + "/topology"].as_string();
      }
    }
  }

  const std::string assoc_str =
      dataset.child(0)["fields/" + field + "/association"].as_string();

  std::unique_ptr<Topology> t =
      topologyFactory(topo_name, dataset.child(domain));
  std::array<conduit::float64, 3> aloc;
  if(assoc_str == "vertex")
  {
    aloc = t->vertex_location(index);
  }
  else if(assoc_str == "element")
  {
    aloc = t->element_location(index);
  }
  else
  {
    ASCENT_ERROR("Location for " << assoc_str << " not implemented");
  }
  conduit::Node loc;
  loc.set(aloc.data(), 3);

  int rank = 0;
  conduit::Node res;
#ifdef ASCENT_MPI_ENABLED
  struct MaxLoc
  {
    double value;
    int rank;
  };

  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);

  MaxLoc maxloc = {max_value, rank};
  MaxLoc maxloc_res;
  MPI_Allreduce(&maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);
  max_value = maxloc.value;

  double *ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, maxloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, maxloc_res.rank, mpi_comm);
  MPI_Bcast(&index, 1, MPI_INT, maxloc_res.rank, mpi_comm);

  loc.set(ploc, 3);
  rank = maxloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["index"] = index;
  res["assoc"] = assoc_str;
  res["position"] = loc;
  res["value"] = max_value;

  return res;
}

conduit::Node
get_state_var(const conduit::Node &dataset, const std::string &var_name)
{
  bool has_state = false;
  conduit::Node state;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_state && dom.has_path("state/" + var_name))
    {
      has_state = true;
      state = dom["state/" + var_name];
    }
  }
  return state;
}

std::string
field_assoc(const conduit::Node &dataset, const std::string &field_name)
{
  bool vertex = true;
  bool rank_has = false;

  const std::string field_path = "fields/" + field_name;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path(field_path))
    {
      rank_has = true;
      std::string asc = dom[field_path + "/association"].as_string();
      if(asc == "element")
      {
        vertex = false;
      }
    }
  }

  bool my_vote = rank_has && vertex;
  bool vertex_vote = global_someone_agrees(my_vote);
  my_vote = rank_has && !vertex;
  bool element_vote = global_someone_agrees(my_vote);

  if(vertex_vote && element_vote)
  {
    ASCENT_ERROR("There is disagreement about the association "
                 << "of field " << field_name);
  }

  return vertex_vote ? "vertex" : "element";
}

std::string
field_type(const conduit::Node &dataset, const std::string &field_name)
{
  bool is_double = true;
  bool rank_has = false;
  bool error = false;

  const std::string field_path = "fields/" + field_name;
  std::string type_name;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path(field_path))
    {
      rank_has = true;
      std::string asc = dom[field_path + "/association"].as_string();
      if(dom[field_path + "/values"].dtype().is_float32())
      {
        is_double = false;
      }
      else if(!dom[field_path + "/values"].dtype().is_float64())
      {
        type_name = dom[field_path + "/values"].dtype().name();
        error = true;
      }
    }
  }

  error = global_agreement(error);
  if(error)
  {

    ASCENT_ERROR("Field '" << field_name << "' is neither float or double."
                           << " type is '" << type_name << "'."
                           << " Contact someone.");
  }

  bool my_vote = rank_has && is_double;
  bool double_vote = global_someone_agrees(my_vote);

  return double_vote ? "double" : "float";
}

void
topology_types(const conduit::Node &dataset,
               const std::string &topo_name,
               int topo_types[5])
{

  for(int i = 0; i < 5; ++i)
  {
    topo_types[i] = 0;
  }

  const int num_domains = dataset.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = dataset.child(0);
    if(dom.has_path("topologies/" + topo_name))
    {
      const std::string topo_type =
          dom["topologies/" + topo_name + "/type"].as_string();
      if(topo_type == "points")
      {
        topo_types[0] += 1;
      }
      else if(topo_type == "uniform")
      {
        topo_types[1] += 1;
      }
      else if(topo_type == "rectilinear")
      {
        topo_types[2] += 1;
      }
      else if(topo_type == "structured")
      {
        topo_types[3] += 1;
      }
      else if(topo_type == "unstructured")
      {
        topo_types[4] += 1;
      }
    }
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Allreduce(MPI_IN_PLACE, topo_types, 5, MPI_INT, MPI_SUM, mpi_comm);
#endif
}

int
topo_dim(const std::string &topo_name, const conduit::Node &dom)
{
  if(!dom.has_path("topologies/" + topo_name))
  {
    ASCENT_ERROR("Topology '" << topo_name << "' not found in domain.");
  }

  const conduit::Node &n_topo = dom["topologies/" + topo_name];

  const std::string c_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = dom["coordsets/" + c_name];
  const std::string c_type = n_coords["type"].as_string();

  int num_dims;
  if(c_type == "uniform")
  {
    num_dims = n_coords["dims"].number_of_children();
  }
  else if(c_type == "rectilinear" || c_type == "explicit")
  {
    num_dims = n_coords["values"].number_of_children();
  }
  else
  {
    num_dims = -1;
    ASCENT_ERROR("Unknown coordinate set type: '" << c_type << "'.");
  }
  if(num_dims <= 0 || num_dims > 3)
  {
    ASCENT_ERROR("The Architect: topology '"
                 << topo_name << "' with " << num_dims
                 << " dimensions is not supported.");
  }
  return num_dims;
}

int
spatial_dims(const conduit::Node &dataset, const std::string &topo_name)
{
  const int num_domains = dataset.number_of_children();

  bool is_3d = false;
  bool rank_has = false;

  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &domain = dataset.child(i);
    if(!domain.has_path("topologies/" + topo_name))
    {
      continue;
    }

    rank_has = true;
    const conduit::Node &n_topo = domain["topologies/" + topo_name];

    const std::string c_name = n_topo["coordset"].as_string();
    const conduit::Node n_coords = domain["coordsets/" + c_name];
    const std::string c_type = n_coords["type"].as_string();

    if(c_type == "uniform")
    {
      if(n_coords.has_path("dims/k"))
      {
        is_3d = true;
      }
      break;
    }

    if(c_type == "rectilinear" || c_type == "explicit")
    {
      if(n_coords.has_path("values/z"))
      {
        is_3d = true;
      }
      break;
    }
  }

  bool my_vote = rank_has && is_3d;
  bool vote_3d = global_someone_agrees(my_vote);
  my_vote = rank_has && !is_3d;
  bool vote_2d = global_someone_agrees(my_vote);

  if(vote_2d && vote_3d)
  {
    ASCENT_ERROR("There is disagreement about the spatial dims"
                 << "of the topoloy '" << topo_name << "'");
  }

  return vote_3d ? 3 : 2;
}

void
paint_nestsets(const std::string nestset_name,
               conduit::Node &dom,
               conduit::Node &field)
{
  if(!dom.has_path("nestsets/" + nestset_name))
  {
    ASCENT_ERROR("No nestset with that name");
  }

  conduit::Node &nestset = dom["nestsets/" + nestset_name];
  const std::string topo_name = nestset["topology"].as_string();
  const conduit::Node &topo = dom["topologies/" + topo_name];

  if(topo["type"].as_string() == "unstructured")
  {
    ASCENT_ERROR("Paint nestsets: cannot paint on unstructured topology");
  }

  int el_dims[3] = {1, 1, 1};
  bool is_3d = false;

  if(topo["type"].as_string() == "structured")
  {
    el_dims[0] = topo["elements/dims/i"].to_int32();
    el_dims[1] = topo["elements/dims/j"].to_int32();
    if(topo.has_path("elements/dims/k"))
    {
      is_3d = true;
      el_dims[2] = topo["elements/dims/k"].to_int32();
    }
  }
  else
  {
    const std::string coord_name = topo["coordset"].as_string();
    const conduit::Node &coords = dom["coordsets/" + coord_name];
    if(coords["type"].as_string() == "uniform")
    {
      el_dims[0] = coords["dims/i"].as_int32() - 1;
      el_dims[1] = coords["dims/j"].as_int32() - 1;

      if(coords.has_path("dims/k"))
      {
        is_3d = true;
        el_dims[2] = topo["dims/k"].to_int32();
      }
    }
    else if(coords["type"].as_string() == "rectilinear")
    {
      el_dims[0] = coords["values/x"].dtype().number_of_elements() - 1;
      el_dims[1] = coords["values/y"].dtype().number_of_elements() - 1;
      if(coords.has_path("values/z"))
      {
        is_3d = true;
        el_dims[1] = coords["values/z"].dtype().number_of_elements() - 1;
      }
    }
    else
    {
      ASCENT_ERROR("unknown coord type");
    }
  }
  // ok, now paint

  conduit::int32 field_size = el_dims[0] * el_dims[1];
  if(is_3d)
  {
    field_size *= el_dims[2];
  }

  conduit::int32_array levels;
  // check to see if the field already has data or if
  // we need to create a new field
  if(field.has_path("values"))
  {
    const int fsize = field["values"].dtype().number_of_elements();
    if(fsize != field_size)
    {
      ASCENT_ERROR("Paint: field given is allocated, but does not"
                   << " match the expected size " << fsize << " "
                   << field_size);
    }
    levels = field["values"].value();
  }
  else
  {
    field["association"] = "element";
    field["topology"] = topo_name;
    field["values"] = conduit::DataType::int32(field_size);
    levels = field["values"].value();
    for(int i = 0; i < field_size; ++i)
    {
      levels[i] = 0;
    }
  }

  const int windows = nestset["windows"].number_of_children();

  for(int i = 0; i < windows; ++i)
  {
    const conduit::Node &window = nestset["windows"].child(i);
    if(window["domain_type"].as_string() != "child")
    {
      continue;
    }

    int origin[3];
    origin[0] = window["origin/i"].to_int32();
    origin[1] = window["origin/j"].to_int32();

    if(is_3d)
    {
      origin[2] = window["origin/k"].to_int32();
    }

    int dims[3];
    dims[0] = window["dims/i"].to_int32();
    dims[1] = window["dims/j"].to_int32();
    if(is_3d)
    {
      dims[2] = window["dims/k"].to_int32();
    }
    if(is_3d)
    {
      // all the nesting relationship is local
      for(int z = origin[2]; z < origin[2] + dims[2]; ++z)
      {
        const int z_offset = z * el_dims[0] * el_dims[1];
        for(int y = origin[1]; y < origin[1] + dims[1]; ++y)
        {
          const conduit::int32 y_offset = y * el_dims[0];
          for(int x = origin[0]; x < origin[0] + dims[0]; ++x)
          {
            // this might a ghost field, but we only want to
            // mask real zones that are masked by finer grids
            conduit::int32 value = levels[z_offset + y_offset + x];
            if(value == 0)
            {
              levels[z_offset + y_offset + x] = 1;
            }
          }
        }
      }
    }
    else
    {
      // all the nesting relationship is local
      for(int y = origin[1]; y < origin[1] + dims[1]; ++y)
      {
        const conduit::int32 y_offset = y * el_dims[0];
        for(int x = origin[0]; x < origin[0] + dims[0]; ++x)
        {
          // this might a ghost field, but we only want to
          // mask real zones that are masked by finer grids
          conduit::int32 value = levels[y_offset + x];
          if(value == 0)
          {
            levels[y_offset + x] = 1;
          }
        }
      }
    }
  }

} // paint

std::string
field_topology(const conduit::Node &dataset, const std::string &field_name)
{
  std::string topo_name;
  const int num_domains = dataset.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field_name))
    {
      topo_name = dom["fields/" + field_name + "/topology"].as_string();
      break;
    }
  }

#if defined(ASCENT_MPI_ENABLED)
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);

  struct MaxLoc
  {
    double size;
    int rank;
  };

  // there is no MPI_INT_INT so shove the "small" size into double
  MaxLoc maxloc = {(double)topo_name.length(), rank};
  MaxLoc maxloc_res;
  MPI_Allreduce(&maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);

  conduit::Node msg;
  msg["topo"] = topo_name;
  conduit::relay::mpi::broadcast_using_schema(msg, maxloc_res.rank, mpi_comm);

  if(!msg["topo"].dtype().is_string())
  {
    ASCENT_ERROR("failed to broadcast topo name");
  }
  topo_name = msg["topo"].as_string();
#endif
  return topo_name;
}

// double or float for a topology in a given domain
std::string
coord_dtype(const std::string &topo_name, const conduit::Node &domain)
{
  // ok, so we can have a mix of uniform and non-uniform
  // coords, where non-uniform coords have arrays
  // if we only have unirform, the double,
  // if some have arrays, then go with whatever
  // that is.
  bool is_float = false;
  bool has_array = false;
  bool error = false;

  const std::string topo_path = "topologies/" + topo_name;
  std::string type_name;

  if(domain.has_path(topo_path))
  {
    std::string coord_name = domain[topo_path + "/coordset"].as_string();
    const conduit::Node &n_coords = domain["coordsets/" + coord_name];
    const std::string coords_type = n_coords["type"].as_string();
    if(coords_type != "uniform")
    {
      has_array = true;

      if(n_coords["values/x"].dtype().is_float32())
      {
        is_float = true;
      }
      else if(!n_coords["values/x"].dtype().is_float64())
      {
        is_float = false;
        type_name = n_coords["/values/x"].dtype().name();
        error = true;
      }
    }
  }
  else
  {
    ASCENT_ERROR("Could not determine the data type of topology '"
                 << topo_name << "' in domain '" << domain.name()
                 << "' because it was not found there.");
  }

  if(error)
  {

    ASCENT_ERROR("Coords array from topo '" << topo_name
                                            << "' is neither float or double."
                                            << " type is '" << type_name << "'."
                                            << " Contact someone.");
  }

  bool my_vote = has_array && is_float;

  return my_vote ? "float" : "double";
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
