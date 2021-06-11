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
#include "ascent_array.hpp"
#include "ascent_array_utils.hpp"
#include "ascent_dispatch.hpp"
#include "ascent_memory_interface.hpp"
#include "ascent_execution.hpp"
#include "ascent_conduit_reductions.hpp"

#include <ascent_logging.hpp>

#include <ascent_mpi_utils.hpp>


#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

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
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

bool
at_least_one(bool local)
{
  bool agreement = local;
#ifdef ASCENT_MPI_ENABLED
  int local_boolean = local ? 1 : 0;
  int global_count = 0;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_count),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_count > 0)
  {
    agreement = true;
  }
#endif
  return agreement;
}

struct UniformCoords
{
  conduit::float64 m_origin[3] = {0., 0., 0.};
  conduit::float64 m_spacing[3] = {1., 1., 1.};
  int m_dims[3] = {0, 0, 0};
  bool m_is_2d = true;

  UniformCoords(const conduit::Node &n_coords)
  {
    populate(n_coords);
  }

  void
  populate(const conduit::Node &n_coords)
  {

    const conduit::Node &n_dims = n_coords["dims"];

    m_dims[0] = n_dims["i"].to_int();
    m_dims[1] = n_dims["j"].to_int();
    m_dims[2] = 1;

    // check for 3d
    if(n_dims.has_path("k"))
    {
      m_dims[2] = n_dims["k"].to_int();
      m_is_2d = false;
    }

    const conduit::Node &n_origin = n_coords["origin"];

    m_origin[0] = n_origin["x"].to_float64();
    m_origin[1] = n_origin["y"].to_float64();

    if(n_origin.has_child("z"))
    {
      m_origin[2] = n_origin["z"].to_float64();
    }

    const conduit::Node &n_spacing = n_coords["spacing"];

    m_spacing[0] = n_spacing["dx"].to_float64();
    m_spacing[1] = n_spacing["dy"].to_float64();

    if(n_spacing.has_path("dz"))
    {
      m_spacing[2] = n_spacing["dz"].to_float64();
    }
  }
};

int
get_num_indices(const std::string &shape_type)
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
    ASCENT_ERROR("Unsupported element type " << shape_type);
  }
  return num;
}

void
logical_index_2d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = vert_index / dims[0];
}

void
logical_index_3d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = (vert_index / dims[0]) % dims[1];
  idx[2] = vert_index / (dims[0] * dims[1]);
}

void
get_element_indices(const conduit::Node &n_topo,
                    const int index,
                    std::vector<int> &indices)
{

  const std::string mesh_type = n_topo["type"].as_string();
  if(mesh_type == "unstructured")
  {
    // supports only single element type
    const conduit::Node &n_topo_eles = n_topo["elements"];

    // get the shape
    const std::string ele_shape = n_topo_eles["shape"].as_string();
    const int num_indices = get_num_indices(ele_shape);

    indices.resize(num_indices);
    // look up the connectivity
    // this is an array that could be on the GPU, so we have to
    // use special care
    const conduit::Node &n_topo_conn = n_topo_eles["connectivity"];
    MemoryInterface<int> conn(n_topo_eles, "connectivity");
    const int offset = index * num_indices;
    for(int i = 0; i < num_indices; ++i)
    {
      indices[i] = conn.value(offset + i,0);
    }
  }
  else
  {
    bool is_2d = true;
    int vert_dims[3] = {0, 0, 0};
    vert_dims[0] = n_topo["elements/dims/i"].to_int32() + 1;
    vert_dims[1] = n_topo["elements/dims/j"].to_int32() + 1;

    if(n_topo.has_path("elements/dims/k"))
    {
      vert_dims[2] = n_topo["elements/dims/k"].to_int32() + 1;
      is_2d = false;
    }

    const int element_dims[3] = {
        vert_dims[0] - 1, vert_dims[1] - 1, vert_dims[2] - 1};

    int element_index[3] = {0, 0, 0};
    if(is_2d)
    {
      indices.resize(4);
      logical_index_2d(element_index, index, element_dims);

      indices[0] = element_index[1] * vert_dims[0] + element_index[0];
      indices[1] = indices[0] + 1;
      indices[2] = indices[1] + vert_dims[0];
      indices[3] = indices[2] - 1;
    }
    else
    {
      indices.resize(8);
      logical_index_3d(element_index, index, element_dims);

      indices[0] =
          (element_index[2] * vert_dims[1] + element_index[1]) * vert_dims[0] +
          element_index[0];
      indices[1] = indices[0] + 1;
      indices[2] = indices[1] + vert_dims[1];
      indices[3] = indices[2] - 1;
      indices[4] = indices[0] + vert_dims[0] * vert_dims[2];
      indices[5] = indices[4] + 1;
      indices[6] = indices[5] + vert_dims[1];
      indices[7] = indices[6] - 1;
    }
  }
}

conduit::Node
get_uniform_vert(const conduit::Node &n_coords, const int &index)
{

  UniformCoords coords(n_coords);

  int logical_index[3] = {0, 0, 0};

  if(coords.m_is_2d)
  {
    logical_index_2d(logical_index, index, coords.m_dims);
  }
  else
  {
    logical_index_3d(logical_index, index, coords.m_dims);
  }

  double vert[3];
  vert[0] = coords.m_origin[0] + logical_index[0] * coords.m_spacing[0];
  vert[1] = coords.m_origin[1] + logical_index[1] * coords.m_spacing[1];
  vert[2] = coords.m_origin[2] + logical_index[2] * coords.m_spacing[2];

  conduit::Node res;
  res.set(vert, 3);
  return res;
}

conduit::Node
get_explicit_vert(const conduit::Node &n_coords, const int &index)
{
  bool is_float64 = true;
  if(n_coords["values/x"].dtype().is_float32())
  {
    is_float64 = false;
  }
  double vert[3] = {0., 0., 0.};
  if(is_float64)
  {
    MemoryInterface<double> array(n_coords);
    vert[0] = array.value(index,"x");
    vert[1] = array.value(index,"y");
    if(array.components() == 3)
    {
      vert[2] = array.value(index,"z");
    }
  }
  else
  {
    MemoryInterface<float> array(n_coords);
    vert[0] = array.value(index,"x");
    vert[1] = array.value(index,"y");
    if(array.components() == 3)
    {
      vert[2] = array.value(index,"z");
    }
  }

  conduit::Node res;
  res.set(vert, 3);
  return res;
}

conduit::Node
get_rectilinear_vert(const conduit::Node &n_coords, const int &index)
{
  bool is_float64 = true;

  int dims[3] = {0, 0, 0};
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
  double vert[3] = {0., 0., 0.};

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
    MemoryInterface<double> f_coords(n_coords);
    vert[0] = f_coords.value(logical_index[0],"x");
    vert[1] = f_coords.value(logical_index[1],"y");
    if(dims[2] != 0)
    {
      vert[2] = f_coords.value(logical_index[2],"z");
    }
  }
  else
  {
    MemoryInterface<float> f_coords(n_coords);
    vert[0] = f_coords.value(logical_index[0],"x");
    vert[1] = f_coords.value(logical_index[1],"y");
    if(dims[2] != 0)
    {
      vert[2] = f_coords.value(logical_index[2],"z");
    }
  }

  conduit::Node res;
  res.set(vert, 3);
  return res;
}
// ----------------------  element locations ---------------------------------
conduit::Node
get_uniform_element(const conduit::Node &n_coords, const int &index)
{

  UniformCoords coords(n_coords);

  int logical_index[3] = {0, 0, 0};
  const int element_dims[3] = {
      coords.m_dims[0] - 1, coords.m_dims[1] - 1, coords.m_dims[2] - 1};

  if(coords.m_is_2d)
  {
    logical_index_2d(logical_index, index, element_dims);
  }
  else
  {
    logical_index_3d(logical_index, index, element_dims);
  }

  // element logical index will be the lower left point index

  double vert[3] = {0., 0., 0.};
  vert[0] = coords.m_origin[0] + logical_index[0] * coords.m_spacing[0] +
            coords.m_spacing[0] * 0.5;
  vert[1] = coords.m_origin[1] + logical_index[1] * coords.m_spacing[1] +
            coords.m_spacing[1] * 0.5;
  vert[2] = coords.m_origin[2] + logical_index[2] * coords.m_spacing[2] +
            coords.m_spacing[2] * 0.5;

  conduit::Node res;
  res.set(vert, 3);
  return res;
}

conduit::Node
get_rectilinear_element(const conduit::Node &n_coords, const int &index)
{
  bool is_float64 = true;

  int dims[3] = {0, 0, 0};
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
  const int element_dims[3] = {dims[0] - 1, dims[1] - 1, dims[2] - 1};

  double vert[3] = {0., 0., 0.};

  int logical_index[3] = {0, 0, 0};

  if(dims[2] == 0)
  {
    logical_index_2d(logical_index, index, element_dims);
  }
  else
  {
    logical_index_3d(logical_index, index, element_dims);
  }

  if(is_float64)
  {
    conduit::float64_array x_a = n_coords["values/x"].value();
    conduit::float64_array y_a = n_coords["values/y"].value();
    vert[0] = (x_a[logical_index[0]] + x_a[logical_index[0] + 1]) * 0.5;
    vert[1] = (y_a[logical_index[1]] + y_a[logical_index[1] + 1]) * 0.5;
    if(dims[2] != 0)
    {
      conduit::float64_array z_a = n_coords["values/z"].value();
      vert[2] = (z_a[logical_index[2]] + z_a[logical_index[2] + 1]) * 0.5;
    }
  }
  else
  {
    conduit::float32_array x_a = n_coords["values/x"].value();
    conduit::float32_array y_a = n_coords["values/y"].value();
    vert[0] = (x_a[logical_index[0]] + x_a[logical_index[0] + 1]) * 0.5;
    vert[1] = (y_a[logical_index[1]] + y_a[logical_index[1] + 1]) * 0.5;
    if(dims[2] != 0)
    {
      conduit::float32_array z_a = n_coords["values/z"].value();
      vert[2] = (z_a[logical_index[2]] + z_a[logical_index[2] + 1]) * 0.5;
    }
  }

  conduit::Node res;
  res.set(vert, 3);
  return res;
}

conduit::Node
get_explicit_element(const conduit::Node &n_coords,
                     const conduit::Node &n_topo,
                     const int &index)
{
  std::vector<int> conn;
  get_element_indices(n_topo, index, conn);
  const int num_indices = conn.size();
  double vert[3] = {0., 0., 0.};
  for(int i = 0; i < num_indices; ++i)
  {
    int vert_index = conn[i];
    conduit::Node n_vert = get_explicit_vert(n_coords, vert_index);
    double *ptr = n_vert.value();
    vert[0] += ptr[0];
    vert[1] += ptr[1];
    vert[2] += ptr[2];
  }

  vert[0] /= double(num_indices);
  vert[1] /= double(num_indices);
  vert[2] /= double(num_indices);

  conduit::Node res;
  res.set(vert, 3);
  return res;
}


struct VertexFunctor
{
  Array<double> m_vertices;
  int m_dims;

  template<typename MeshType, typename Exec>
  void operator()(MeshType &mesh, const Exec &)
  {
    using fp = typename Exec::for_policy;
    const int size = mesh.m_num_points;
    //const int dims = mesh.m_dims;
    // we cant capture class members
    m_dims = mesh.m_dims;

    // always output 3 components
    m_vertices.resize(size * 3);
    double *verts_ptr = m_vertices.ptr(Exec::memory_space);
    std::cout<<"**** VertexFunctor Mem space "<<Exec::memory_space<<"\n";

    RAJA::forall<fp>
      (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type point_idx)
    {
      double vert[3];
      mesh.vertex(point_idx, vert);
      const int offset = point_idx * 3;
      verts_ptr[offset + 0] = vert[0];
      verts_ptr[offset + 1] = vert[1];
      verts_ptr[offset + 2] = vert[2];
    });
    ASCENT_ERROR_CHECK();
  }
};

struct CentroidFunctor
{
  Array<double> m_centroids;
  int m_dims;

  template<typename MeshType, typename Exec>
  void operator()(MeshType &mesh, const Exec &)
  {
    using fp = typename Exec::for_policy;
    const int size = mesh.m_num_cells;
    const int dims = mesh.m_dims;

    // one component for each dim
    m_centroids.resize(size * mesh.m_dims);
    double *centroids_ptr = m_centroids.ptr(Exec::memory_space);
    std::cout<<"Mem space "<<Exec::memory_space<<"\n";

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type cell_idx)
    {
      centroids_ptr[cell_idx] = cell_idx;
      int indices[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
      mesh.cell_indices(cell_idx, indices);

      double centroid[3] = {0., 0., 0.};
      const int num_indices = mesh.m_num_indices;
      for(int i = 0; i < num_indices; ++i)
      {
        double vert[3];
        mesh.vertex(indices[i], vert);
        for(int c = 0; c < 3; c++)
        {
          centroid[c] += vert[c];
        }
      }

      centroid[0] /= double(num_indices);
      centroid[1] /= double(num_indices);
      centroid[2] /= double(num_indices);

      const int out_offset = cell_idx * dims;
      for(int i = 0; i < dims; i++)
      {
        centroids_ptr[out_offset + i] = centroid[i];
      }

    });
    ASCENT_ERROR_CHECK();
  }
};

//-----------------------------------------------------------------------------
}; // namespace detail
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

conduit::Node
vert_location(const conduit::Node &domain,
              const int &index,
              const std::string &topo_name)
{
  std::string topo = topo_name;
  // if we don't specify a topology, find the first topology ...
  if(topo_name == "")
  {
    conduit::NodeConstIterator itr = domain["topologies"].children();
    itr.next();
    topo = itr.name();
  }

  const conduit::Node &n_topo = domain["topologies"][topo];
  const std::string mesh_type = n_topo["type"].as_string();
  const std::string coords_name = n_topo["coordset"].as_string();

  const conduit::Node &n_coords = domain["coordsets"][coords_name];

  conduit::Node res;
  if(mesh_type == "uniform")
  {
    res = detail::get_uniform_vert(n_coords, index);
  }
  else if(mesh_type == "rectilinear")
  {
    res = detail::get_rectilinear_vert(n_coords, index);
  }
  else if(mesh_type == "unstructured" || mesh_type == "structured")
  {
    res = detail::get_explicit_vert(n_coords, index);
  }
  else
  {
    ASCENT_ERROR("The Architect: unknown mesh type: '" << mesh_type << "'");
  }

  return res;
}

conduit::Node
element_location(const conduit::Node &domain,
                 const int &index,
                 const std::string &topo_name)
{
  std::string topo = topo_name;
  // if we don't specify a topology, find the first topology ...
  if(topo_name == "")
  {
    conduit::NodeConstIterator itr = domain["topologies"].children();
    itr.next();
    topo = itr.name();
  }

  const conduit::Node &n_topo = domain["topologies"][topo];
  const std::string mesh_type = n_topo["type"].as_string();
  const std::string coords_name = n_topo["coordset"].as_string();

  const conduit::Node &n_coords = domain["coordsets"][coords_name];

  conduit::Node res;
  if(mesh_type == "uniform")
  {
    res = detail::get_uniform_element(n_coords, index);
  }
  else if(mesh_type == "rectilinear")
  {
    res = detail::get_rectilinear_element(n_coords, index);
  }
  else if(mesh_type == "unstructured" || mesh_type == "structured")
  {
    res = detail::get_explicit_element(n_coords, n_topo, index);
  }
  else
  {
    ASCENT_ERROR("The Architect: unknown mesh type: '" << mesh_type << "'");
  }

  return res;
}

std::string
possible_components(const conduit::Node &dataset,
                   const std::string &field_name)
{
  std::string res;
  if(dataset.number_of_children() > 0)
  {
    const conduit::Node &dom = dataset.child(0);
    if(dom.has_path("fields/" + field_name))
    {
      if(dom["fields/"+field_name+"/values"].number_of_children() > 0)
      {
        std::vector<std::string> names
          = dom["fields/"+field_name+"/values"].child_names();
        std::stringstream ss;
        ss<<"[";
        for(auto name : names)
        {
          ss<<" '"<<name<<"'";
        }
        ss<<"]";
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
      if(num_children == 0 || num_children == 1)
      {
        is_scalar = true;
      }
    }
  }
  // check to see if the field exists in any rank
  is_scalar = detail::at_least_one(is_scalar);
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
  has_field = detail::at_least_one(has_field);
  return has_field;
}

bool
has_component(const conduit::Node &dataset,
              const std::string &field_name,
              const std::string &component)
{

  bool has_comp= false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_comp && dom.has_path("fields/" + field_name + "/values/"+component))
    {
      has_comp = true;
    }
  }
  // check to see if the field exists in any rank
  has_comp= detail::at_least_one(has_comp);
  return has_comp;
}

// TODO If someone names their fields x,y,z things will go wrong
bool
is_xyz(const std::string &axis_name)
{
  return axis_name == "x" || axis_name == "y" || axis_name == "z";
}

int
num_points(const conduit::Node &domain, const std::string &topo_name)
{
  int res = 0;

  const conduit::Node &n_topo = domain["topologies/" + topo_name];

  const std::string c_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = domain["coordsets/" + c_name];
  const std::string c_type = n_coords["type"].as_string();

  if(c_type == "uniform")
  {
    res = n_coords["dims/i"].to_int32();
    if(n_coords.has_path("dims/j"))
    {
      res *= n_coords["dims/j"].to_int32();
    }
    if(n_coords.has_path("dims/k"))
    {
      res *= n_coords["dims/k"].to_int32();
    }
  }

  if(c_type == "rectilinear")
  {
    res = n_coords["values/x"].dtype().number_of_elements();

    if(n_coords.has_path("values/y"))
    {
      res *= n_coords["values/y"].dtype().number_of_elements();
    }

    if(n_coords.has_path("values/z"))
    {
      res *= n_coords["values/z"].dtype().number_of_elements();
    }
  }

  if(c_type == "explicit")
  {
    res = n_coords["values/x"].dtype().number_of_elements();
  }

  return res;
}

int
num_cells(const conduit::Node &domain, const std::string &topo_name)
{
  const conduit::Node &n_topo = domain["topologies/" + topo_name];
  const std::string topo_type = n_topo["type"].as_string();

  int res = -1;

  if(topo_type == "unstructured")
  {
    const std::string shape = n_topo["elements/shape"].as_string();
    const int conn_size =
        n_topo["elements/connectivity"].dtype().number_of_elements();
    const int per_cell = detail::get_num_indices(shape);
    res = conn_size / per_cell;
  }

  if(topo_type == "points")
  {
    return num_points(domain, topo_name);
  }

  const std::string c_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = domain["coordsets/" + c_name];

  if(topo_type == "uniform")
  {
    res = n_coords["dims/i"].to_int32() - 1;
    if(n_coords.has_path("dims/j"))
    {
      res *= n_coords["dims/j"].to_int32() - 1;
    }
    if(n_coords.has_path("dims/k"))
    {
      res *= n_coords["dims/k"].to_int32() - 1;
    }
  }

  if(topo_type == "rectilinear")
  {
    res = n_coords["values/x"].dtype().number_of_elements() - 1;

    if(n_coords.has_path("values/y"))
    {
      res *= n_coords["values/y"].dtype().number_of_elements() - 1;
    }

    if(n_coords.has_path("values/z"))
    {
      res *= n_coords["values/z"].dtype().number_of_elements() - 1;
    }
  }

  if(topo_type == "structured")
  {
    res = n_topo["elements/dims/i"].to_int32() - 1;
    if(n_topo.has_path("elements/dims/j"))
    {
      res *= n_topo["elements/dims/j"].to_int32() - 1;
    }
    if(n_topo.has_path("elements/dims/k"))
    {
      res *= n_topo["elements/dims/k"].to_int32() - 1;
    }
  }

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
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_histogram(dom[path], min_val, max_val, num_bins);

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
    if(!dom.has_path("topologies/"+topo_name))
    {
      continue;
    }
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
            min_coords[i], field_reduction_min(n_coords, axes[i][0])["value"].as_double());
        max_coords[i] = std::max(
            max_coords[i], field_reduction_max(n_coords, axes[i][0])["value"].as_double());
        //min_coords[i] = std::min(
        //    min_coords[i], field_reduction_min(n_coords[axis_path])["value"].as_double());
        //max_coords[i] = std::max(
        //    max_coords[i], field_reduction_max(n_coords[axis_path])["value"].as_double());
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

// get the association and topology and ensure they are the same
conduit::Node
global_topo_and_assoc(const conduit::Node &dataset,
                      const std::vector<std::string> var_names)
{
  std::string assoc_str;
  std::string topo_name;
  bool error = false;
  conduit::Node error_msg;

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
          error_msg.append() = "All Binning fields must have the same association.";
          error = true;
        }

        const std::string cur_topo_name =
            dom["fields/" + var_name + "/topology"].as_string();
        if(topo_name.empty())
        {
          topo_name = cur_topo_name;
        }
        else if(topo_name != cur_topo_name)
        {
          error_msg.append() = "All Binning fields must have the same topology.";
          error = true;
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


  if(assoc_str != "" && assoc_str != msg["assoc_str"].as_string())
  {
    error_msg.append() = "All Binning fields must have the same association.";
    error = true;
  }
  if(topo_name != "" && topo_name != msg["topo_name"].as_string())
  {
    error_msg.append() = "All Binning fields must have the same topology.";
    error = true;
  }
  assoc_str = msg["assoc_str"].as_string();
  topo_name = msg["topo_name"].as_string();
#endif
  if(assoc_str.empty())
  {

    error_msg.append() = "Could not determine the associate from the given "
                          "reduction_var and axes. Try supplying a field.";
    error = true;
  }
  else if(assoc_str != "vertex" && assoc_str != "element")
  {
    error_msg.append() = "Unknown association: '"
                       + assoc_str
                       + "'. Binning only supports vertex and element association.";
  }

#ifdef ASCENT_MPI_ENABLED
  int error_int = error ? 1 : 0;
  int global_error;
  MPI_Allreduce(&error_int, &global_error, 1, MPI_INT, MPI_MAX, mpi_comm);
  conduit::Node global_msg;
  error = global_error == 1;
  if(global_error == 1)
  {
    conduit::relay::mpi::all_gather_using_schema(error_msg, global_msg, mpi_comm);
    error_msg = global_msg;
  }
#endif

  if(error)
  {
    ASCENT_ERROR(error_msg.to_yaml());
  }

  conduit::Node res;
  res["topo_name"] = topo_name;
  res["assoc_str"] = assoc_str;
  return res;
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

  // Calculate the size of homes
  conduit::index_t homes_size = 0;
  if(assoc_str == "vertex")
  {
    homes_size = num_points(dom, topo_name);
  }
  else if(assoc_str == "element")
  {
    homes_size = num_cells(dom, topo_name);
  }

  // each domain has a homes array
  // homes maps each datapoint (or cell) to an index in bins
  res.set(conduit::DataType::c_int(homes_size));
  int *homes = res.value();
  for(int i = 0; i < homes_size; ++i)
  {
    homes[i] = 0;
  }

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
        const conduit::float32_array values = dom[values_path].value();
        for(int i = 0; i < values.number_of_elements(); ++i)
        {
          const int bin_index = get_bin_index(values[i], axis);
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
        const conduit::float64_array values = dom[values_path].value();
        for(int i = 0; i < values.number_of_elements(); ++i)
        {
          const int bin_index = get_bin_index(values[i], axis);
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
    else if(is_xyz(axis_name))
    {
      int coord = axis_name[0] - 'x';
      for(int i = 0; i < homes_size; ++i)
      {
        conduit::Node n_loc;
        if(assoc_str == "vertex")
        {
          n_loc = vert_location(dom, i, topo_name);
        }
        else if(assoc_str == "element")
        {
          n_loc = element_location(dom, i, topo_name);
        }
        const double *loc = n_loc.value();
        const int bin_index = get_bin_index(loc[coord], axis);
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
}

void
update_bin(double *bins,
           const int i,
           const double value,
           const std::string &reduction_op)
{
  if(reduction_op == "min")
  {
    bins[i] = std::min(bins[i], value);
  }
  else if(reduction_op == "max")
  {
    bins[i] = std::max(bins[i], value);
  }
  else if(reduction_op == "avg" || reduction_op == "sum" ||
          reduction_op == "pdf")
  {
    bins[2 * i] += value;
    bins[2 * i + 1] += 1;
  }
  else if(reduction_op == "rms")
  {
    bins[2 * i] += value * value;
    bins[2 * i + 1] += 1;
  }
  else if(reduction_op == "var" || reduction_op == "std")
  {
    bins[3 * i] += value * value;
    bins[3 * i + 1] += value;
    bins[3 * i + 2] += 1;
  }
}

void init_bins(Array<double> &bins,
               const std::string reduction_op)
{
  double init_val = 0.;
  if(reduction_op == "max")
  {
    init_val = std::numeric_limits<double>::lowest();
  }
  else
  {
    init_val = std::numeric_limits<double>::max();
  }

  array_memset(bins, init_val);

}

void init_bins(double *bins,
               const int size,
               const std::string reduction_op)
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

int coord_axis_to_int(const std::string coord)
{
  int axis = -1;
  if(coord == "x")
  {
    axis = 0;
  }
  else if(coord == "y")
  {
    axis = 1;
  }
  else if(coord == "z")
  {
    axis = 2;
  }
  if(axis == -1)
  {
    ASCENT_ERROR("Unknown coord axis '"<<coord
                 <<"'. This should not happen");
  }
  return axis;
}
// Create a set of explicit bins for all axes
conduit::Node
create_bins_axes(conduit::Node &bin_axes,
                 const conduit::Node &dataset,
                 std::string topo_name)
{

  // we are most likely to have some spatial bins so just calculate
  // the global bounds
  const conduit::Node &bounds = global_bounds(dataset, topo_name);
  const double *min_coords = bounds["min_coords"].value();
  const double *max_coords = bounds["max_coords"].value();

  bin_axes.print();
  conduit::Node res;

  const int num_axes = bin_axes.number_of_children();
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = bin_axes.child(i);
    const std::string axis_name = axis.name();
    std::cout<<"Axis name "<<axis_name<<"\n";
    if(axis.has_path("bins"))
    {
      // we have explicit bins, so just use them
      // TODO: we could check and sort the bins
      // just in case there was a typo
      res[axis_name] = axis;
      continue;
    }

    // ok, we need to create a uniform binning for this
    // axis. So figure out the mins and maxs
    if(!axis.has_path("num_bins"))
    {
      ASCENT_ERROR("Axis missing num_bins and explicit bins");
    }

    const int num_bins = axis["num_bins"].to_int32();
    if(num_bins < 1)
    {
      ASCENT_ERROR("Binning: There must be at least one bin for axis '"<<
                   axis_name<<"'");
    }

    double min_val, max_val;
    bool has_min = axis.has_path("min_val");
    bool has_max = axis.has_path("max_val");
    if(has_min)
    {
      min_val = axis["min_val"].as_float64();
    }
    if(has_min)
    {
      max_val = axis["max_val"].as_float64();
    }

    if(is_xyz(axis_name))
    {
      int axis_id = coord_axis_to_int(axis_name);
      if(!has_min)
      {
        min_val = min_coords[axis_id];
      }
      if(!has_min)
      {
        max_val = max_coords[axis_id];
      }
      std::cout<<"spatial axis "<<axis_id<<"\n";
    }
    else
    {
      // this is a field, so
      std::cout<<"filed\n";
      if(!has_min)
      {
        min_val = field_min(dataset, axis_name)["value"].as_float64();
      }
      if(!has_max)
      {
        max_val = field_max(dataset, axis_name)["value"].as_float64();
      }
    }

    if(min_val > max_val)
    {
      ASCENT_ERROR("Binning: axis '"<<axis_name<<"' has invalid bounds "
                   <<"min "<<min_val<<" max "<<max_val);
    }

    const int divs = num_bins + 1;
    res[axis_name +"/bins"].set(conduit::DataType::float64(divs));
    double *bins = res[axis_name +"/bins"].value();
    const double delta = (max_val - min_val) / double(num_bins);
    bins[0] = min_val;

    for(int j = 1; j < divs - 1; ++j)
    {
      bins[j] = min_val + j * delta;
    }


  }
  res.print();
  return res;
}

// reduction_op: sum, min, max, avg, pdf, std, var, rms
conduit::Node
binning2(const conduit::Node &dataset,
         conduit::Node &bin_axes,
         const std::string &reduction_var,
         const std::string &reduction_op,
         const double empty_bin_val,
         const std::string &component)
{
  bin_axes.print();

  // first verify that all variables have matching associations
  // and are part of the same topology
  std::vector<std::string> var_names = bin_axes.child_names();
  if(!reduction_var.empty())
  {
    var_names.push_back(reduction_var);
  }
  const conduit::Node &topo_and_assoc =
      global_topo_and_assoc(dataset, var_names);
  const std::string topo_name = topo_and_assoc["topo_name"].as_string();
  const std::string assoc_str = topo_and_assoc["assoc_str"].as_string();

  std::cout<<"Creating BINS\n";
  conduit::Node axes = create_bins_axes(bin_axes, dataset, topo_name);
  std::cout<<"DONE BINS\n";

  const int num_axes = axes.number_of_children();

  // allocate memory for the total number of bins
  size_t num_bins = 1;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    num_bins *= axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;
  }
  std::cout<<"Total bins "<<num_bins<<"\n";

  // we might need additional space to keep track of statistics,
  // i.e., we might need to keep track of the bin sum and counts for
  // average
  int num_bin_vars = 2;
  if(reduction_op == "var" || reduction_op == "std")
  {
    num_bin_vars = 3;
  }
  else if(reduction_op == "min" || reduction_op == "max")
  {
    num_bin_vars = 1;
  }

  const int bins_size = num_bins * num_bin_vars;
  Array<double> bins;
  bins.resize(bins_size);

  std::cout<<"total bins allocation "<<bins_size<<"\n";

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    if(!dom.has_path("topologies/"+topo_name))
    {
      continue;
    }
    centroids(dom, topo_name);
  }
  init_bins(bins, reduction_op);
  return axes;

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    if(!dom.has_path("topologies/"+topo_name))
    {
      continue;
    }
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
        continue;
      }
    }

    conduit::Node n_homes;
    populate_homes(dom, bin_axes, topo_name, assoc_str, n_homes);

    //if(n_homes.has_path("error"))
    //{
    //  ASCENT_INFO("Binning: not binning domain "
    //              << dom_index << " because field: '"
    //              << n_homes["error/field_name"].to_string()
    //              << "' was not found.");
    //  continue;
    //}
    //const int *homes = n_homes.as_int_ptr();
    //const int homes_size = n_homes.dtype().number_of_elements();

    //// update bins
    //if(reduction_var.empty())
    //{
    //  for(int i = 0; i < homes_size; ++i)
    //  {
    //    if(homes[i] != -1)
    //    {
    //      update_bin(bins, homes[i], 1, reduction_op);
    //    }
    //  }
    //}
    //else if(dom.has_path("fields/" + reduction_var))
    //{
    //  const std::string comp_path = component == "" ? "" : "/" + component;
    //  const std::string values_path
    //    = "fields/" + reduction_var + "/values" + comp_path;

    //  if(dom[values_path].dtype().is_float32())
    //  {
    //    const conduit::float32_array values = dom[values_path].value();
    //    for(int i = 0; i < homes_size; ++i)
    //    {
    //      if(homes[i] != -1)
    //      {
    //        update_bin(bins, homes[i], values[i], reduction_op);
    //      }
    //    }
    //  }
    //  else
    //  {
    //    const conduit::float64_array values = dom[values_path].value();
    //    for(int i = 0; i < homes_size; ++i)
    //    {
    //      if(homes[i] != -1)
    //      {
    //        update_bin(bins, homes[i], values[i], reduction_op);
    //      }
    //    }
    //  }
    //}
    //else if(is_xyz(reduction_var))
    //{
    //  int coord = reduction_var[0] - 'x';
    //  for(int i = 0; i < homes_size; ++i)
    //  {
    //    conduit::Node n_loc;
    //    if(assoc_str == "vertex")
    //    {
    //      n_loc = vert_location(dom, i, topo_name);
    //    }
    //    else if(assoc_str == "element")
    //    {
    //      n_loc = element_location(dom, i, topo_name);
    //    }
    //    const double *loc = n_loc.value();
    //    if(homes[i] != -1)
    //    {
    //      update_bin(bins, homes[i], loc[coord], reduction_op);
    //    }
    //  }
    //}
    //else
    //{
    //  ASCENT_INFO("Binning: not binning domain "
    //              << dom_index << " because field: '" << reduction_var
    //              << "' was not found.");
    //}
  }
#if 0

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
    MPI_Allreduce(bins, global_bins, bins_size, MPI_DOUBLE, MPI_MIN, mpi_comm);
  }
  else if(reduction_op == "max")
  {
    MPI_Allreduce(bins, global_bins, bins_size, MPI_DOUBLE, MPI_MAX, mpi_comm);
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
    for(int i = 0; i < num_bins; ++i)
    {
      total += bins[2 * i];
    }
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[2 * i + 1] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[2 * i] / total;
      }
    }
  }
  else if(reduction_op == "min")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[i] == std::numeric_limits<double>::max())
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i];
      }
    }
  }
  else if(reduction_op == "max")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[i] == std::numeric_limits<double>::lowest())
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i];
      }
    }

  }
  else if(reduction_op == "sum")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[2 * i + 1] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[2 * i];
      }
    }
  }
  else if(reduction_op == "avg")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[2 * i];
      const double n = bins[2 * i + 1];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[2 * i];
      const double n = bins[2 * i + 1];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[3 * i];
      const double sumX = bins[3 * i + 1];
      const double n = bins[3 * i + 2];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[3 * i];
      const double sumX = bins[3 * i + 1];
      const double n = bins[3 * i + 2];
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
#endif
}

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
  else if(reduction_op == "min" || reduction_op == "max")
  {
    num_bin_vars = 1;
  }
  const int bins_size = num_bins * num_bin_vars;
  double *bins = new double[bins_size]();
  init_bins(bins, bins_size, reduction_op);

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    if(!dom.has_path("topologies/"+topo_name))
    {
      continue;
    }

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
          update_bin(bins, homes[i], 1, reduction_op);
        }
      }
    }
    else if(dom.has_path("fields/" + reduction_var))
    {
      const std::string comp_path = component == "" ? "" : "/" + component;
      const std::string values_path
        = "fields/" + reduction_var + "/values" + comp_path;

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
            update_bin(bins, homes[i], values[i], reduction_op);
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
            update_bin(bins, homes[i], values[i], reduction_op);
          }
        }
      }
    }
    else if(is_xyz(reduction_var))
    {
      int coord = reduction_var[0] - 'x';
//#ifdef ASCENT_USE_OPENMP
//#pragma omp parallel for
//#endif
      for(int i = 0; i < homes_size; ++i)
      {
        conduit::Node n_loc;
        if(assoc_str == "vertex")
        {
          n_loc = vert_location(dom, i, topo_name);
        }
        else if(assoc_str == "element")
        {
          n_loc = element_location(dom, i, topo_name);
        }
        const double *loc = n_loc.value();
        if(homes[i] != -1)
        {
          update_bin(bins, homes[i], loc[coord], reduction_op);
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
    MPI_Allreduce(bins, global_bins, bins_size, MPI_DOUBLE, MPI_MIN, mpi_comm);
  }
  else if(reduction_op == "max")
  {
    MPI_Allreduce(bins, global_bins, bins_size, MPI_DOUBLE, MPI_MAX, mpi_comm);
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
    for(int i = 0; i < num_bins; ++i)
    {
      total += bins[2 * i];
    }
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[2 * i + 1] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[2 * i] / total;
      }
    }
  }
  else if(reduction_op == "min")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[i] == std::numeric_limits<double>::max())
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i];
      }
    }
  }
  else if(reduction_op == "max")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[i] == std::numeric_limits<double>::lowest())
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[i];
      }
    }

  }
  else if(reduction_op == "sum")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      if(bins[2 * i + 1] == 0)
      {
        res_bins[i] = empty_bin_val;
      }
      else
      {
        res_bins[i] = bins[2 * i];
      }
    }
  }
  else if(reduction_op == "avg")
  {
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[2 * i];
      const double n = bins[2 * i + 1];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX = bins[2 * i];
      const double n = bins[2 * i + 1];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[3 * i];
      const double sumX = bins[3 * i + 1];
      const double n = bins[3 * i + 2];
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
    for(int i = 0; i < num_bins; ++i)
    {
      const double sumX2 = bins[3 * i];
      const double sumX = bins[3 * i + 1];
      const double n = bins[3 * i + 2];
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

void
paint_binning(const conduit::Node &binning, conduit::Node &dataset)
{
  const conduit::Node &bin_axes = binning["attrs/bin_axes/value"];

  // get assoc_str and topo_name
  std::vector<std::string> axis_names = bin_axes.child_names();
  bool all_xyz = true;
  for(const std::string &axis_name : axis_names)
  {
    all_xyz &= is_xyz(axis_name);
  }
  std::string topo_name;
  std::string assoc_str;
  if(all_xyz)
  {
    // pick the first topology from the first domain and use the association
    // from the binning
    topo_name = dataset.child(0)["topologies"].child(0).name();
    assoc_str = binning["attrs/association/value"].as_string();
  }
  else
  {
    const conduit::Node &topo_and_assoc =
        global_topo_and_assoc(dataset, axis_names);
    topo_name = topo_and_assoc["topo_name"].as_string();
    assoc_str = topo_and_assoc["assoc_str"].as_string();
  }

  const double *bins = binning["attrs/value/value"].as_double_ptr();

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    conduit::Node &dom = dataset.child(dom_index);

    conduit::Node n_homes;
    populate_homes(dom, bin_axes, topo_name, assoc_str, n_homes);
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

    std::string reduction_var =
        binning["attrs/reduction_var/value"].as_string();
    if(reduction_var.empty())
    {
      reduction_var = "cnt";
    }
    const std::string field_name =
        "painted_" + reduction_var + "_" +
        binning["attrs/reduction_op/value"].as_string();
    dom["fields/" + field_name + "/association"] = assoc_str;
    dom["fields/" + field_name + "/topology"] = topo_name;
    dom["fields/" + field_name + "/values"].set(
        conduit::DataType::float64(homes_size));
    conduit::float64_array values =
        dom["fields/" + field_name + "/values"].value();
#ifdef ASCENT_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < homes_size; ++i)
    {
      values[i] = bins[homes[i]];
    }
  }

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh", dataset, info))
  {
    info.print();
    ASCENT_ERROR(
        "Failed to verify mesh after painting binning back on the mesh.");
  }
}

void
binning_mesh2(const conduit::Node &binning, conduit::Node &mesh)
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
  mesh["coordsets/binning_coords/type"] = "rectilinear";
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = binning["attrs/bin_axes/value"].child(i);
    if(axis.has_path("bins"))
    {
      // rectilinear
      mesh["coordsets/binning_coords/values/" + axes[i][0]] = axis["bins"];
    }
    else
    {
      // uniform
      const int dim = axis["num_bins"].as_int32() + 1;
      const double delta =
          (axis["max_val"].to_float64() - axis["min_val"].to_float64()) /
          (dim - 1);
      mesh["coordsets/binning_coords/values/" + axes[i][0]].set(
          conduit::DataType::c_double(dim));
      double *bins =
          mesh["coordsets/binning_coords/values/" + axes[i][0]].value();
      for(int j = 0; j < dim; ++j)
      {
        bins[j] = axis["min_val"].to_float64() + j * delta;
      }
    }
  }

  // create topology
  mesh["topologies/binning_topo/type"] = "rectilinear";
  mesh["topologies/binning_topo/coordset"] = "binning_coords";

  // create field
  std::string reduction_var = binning["attrs/reduction_var/value"].as_string();
  if(reduction_var.empty())
  {
    reduction_var = "cnt";
  }
  const std::string field_name =
      reduction_var + "_" + binning["attrs/reduction_op/value"].as_string();
  mesh["fields/" + field_name + "/association"] = "element";
  mesh["fields/" + field_name + "/topology"] = "binning_topo";
  mesh["fields/" + field_name + "/values"].set(binning["attrs/value/value"]);

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh", mesh, info))
  {
    info.print();
    ASCENT_ERROR("Failed to create valid binning mesh.");
  }
}

void
binning_mesh(const conduit::Node &binning, conduit::Node &mesh)
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
  mesh["coordsets/binning_coords/type"] = "rectilinear";
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = binning["attrs/bin_axes/value"].child(i);
    if(axis.has_path("bins"))
    {
      // rectilinear
      mesh["coordsets/binning_coords/values/" + axes[i][0]] = axis["bins"];
    }
    else
    {
      // uniform
      const int dim = axis["num_bins"].as_int32() + 1;
      const double delta =
          (axis["max_val"].to_float64() - axis["min_val"].to_float64()) /
          (dim - 1);
      mesh["coordsets/binning_coords/values/" + axes[i][0]].set(
          conduit::DataType::c_double(dim));
      double *bins =
          mesh["coordsets/binning_coords/values/" + axes[i][0]].value();
      for(int j = 0; j < dim; ++j)
      {
        bins[j] = axis["min_val"].to_float64() + j * delta;
      }
    }
  }

  // create topology
  mesh["topologies/binning_topo/type"] = "rectilinear";
  mesh["topologies/binning_topo/coordset"] = "binning_coords";

  // create field
  std::string reduction_var = binning["attrs/reduction_var/value"].as_string();
  if(reduction_var.empty())
  {
    reduction_var = "cnt";
  }
  const std::string field_name =
      reduction_var + "_" + binning["attrs/reduction_op/value"].as_string();
  mesh["fields/" + field_name + "/association"] = "element";
  mesh["fields/" + field_name + "/topology"] = "binning_topo";
  mesh["fields/" + field_name + "/values"].set(binning["attrs/value/value"]);

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh", mesh, info))
  {
    info.print();
    ASCENT_ERROR("Failed to create valid binning mesh.");
  }
}

conduit::Node
field_entropy(const conduit::Node &hist)
{
  const double *hist_bins = hist["attrs/value/value"].value();
  const int num_bins = hist["attrs/num_bins/value"].to_int32();
  std::string exec = ExecutionManager::preferred_cpu_device();
  double sum = array_sum(hist["attrs/value/value"], exec)["value"].to_float64();
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

  std::string exec = ExecutionManager::preferred_cpu_device();
  double sum = array_sum(hist["attrs/value/value"], exec)["value"].to_float64();

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

  std::string exec = ExecutionManager::preferred_cpu_device();
  double sum = array_sum(hist["attrs/value/value"],exec)["value"].to_float64();

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
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_nan_count(dom[path]);
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
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_inf_count(dom[path]);
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

  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_min(dom[path]);
      double a_min = res["value"].to_float64();
      if(a_min < min_value)
      {
        min_value = a_min;
        index = res["index"].to_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
      }
    }
  }

  // create a default location so everyone has something
  // it won't matter since this rank cant have the value
  double vert[3] = {0., 0., 0.};
  conduit::Node loc;
  loc.set(vert,3);
  std::string assoc_str = "";

  if(domain != -1)
  {
    assoc_str = dataset.child(domain)["fields/" + field + "/association"].as_string();

    if(assoc_str == "vertex")
    {
      loc = vert_location(dataset.child(domain), index);
    }
    else if(assoc_str == "element")
    {
      loc = element_location(dataset.child(domain), index);
    }
    else
    {
      ASCENT_ERROR("Location for " << assoc_str << " not implemented");
    }
  }

  int rank = 0;
  conduit::Node res;
  int assoc_int = assoc_str == "vertex" ? 1 : 0;

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
  min_value = minloc_res.value;

  double *ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, minloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, minloc_res.rank, mpi_comm);
  MPI_Bcast(&index, 1, MPI_INT, minloc_res.rank, mpi_comm);

  // make sure everyone has the same assoc even if they
  // didn't have the data
  MPI_Bcast(&assoc_int, 1, MPI_INT, minloc_res.rank, mpi_comm);

  loc.set(ploc, 3);

  rank = minloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["index"] = index;
  res["assoc"] = assoc_int == 1 ? "vertex" : "element";
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
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_sum(dom[path]);

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

  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/" + field))
    {
      //const std::string path = "fields/" + field + "/values";
      const std::string path = "fields/" + field;
      conduit::Node res;
      res = field_reduction_max(dom[path]);
      double a_max = res["value"].to_float64();
      if(a_max > max_value)
      {
        max_value = a_max;
        index = res["index"].to_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
      }
    }
  }

  double vert[3] = {0., 0., 0.};
  conduit::Node loc;
  loc.set(vert,3);
  std::string assoc_str = "";

  if(domain != -1)
  {
    const std::string assoc_str =
        dataset.child(domain)["fields/" + field + "/association"].as_string();

    if(assoc_str == "vertex")
    {
      loc = vert_location(dataset.child(domain), index);
    }
    else if(assoc_str == "element")
    {
      loc = element_location(dataset.child(domain), index);
    }
    else
    {
      ASCENT_ERROR("Location for " << assoc_str << " not implemented");
    }
  }

  int rank = 0;
  conduit::Node res;
  int assoc_int = assoc_str == "vertex" ? 1 : 0;

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
  max_value = maxloc_res.value;

  double *ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, maxloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, maxloc_res.rank, mpi_comm);
  MPI_Bcast(&index, 1, MPI_INT, maxloc_res.rank, mpi_comm);

  // make sure everyone has the same assoc even if they
  // didn't have the data
  MPI_Bcast(&assoc_int, 1, MPI_INT, maxloc_res.rank, mpi_comm);

  loc.set(ploc, 3);
  rank = maxloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["index"] = index;
  res["assoc"] = assoc_int == 1 ? "vertex" : "element";
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

void paint_nestsets(const std::string nestset_name,
                    const std::string topo_name,
                    conduit::Node &dom,
                    conduit::Node &field)
{
  const conduit::Node &topo = dom["topologies/"+topo_name];

  if(topo["type"].as_string() == "unstructured")
  {
    ASCENT_ERROR("Paint nestsets: cannot paint on unstructured topology");
  }

  int el_dims[3] = {1,1,1};
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
    const conduit::Node &coords = dom["coordsets/"+coord_name];
    if(coords["type"].as_string() == "uniform")
    {
      el_dims[0] = coords["dims/i"].as_int32() - 1;
      el_dims[1] = coords["dims/j"].as_int32() - 1;

      if(coords.has_path("dims/k"))
      {
        is_3d = true;
        el_dims[2] = coords["dims/k"].to_int32() -1;
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
                   <<" match the expected size "<<fsize<<" "<<field_size);
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
  // its possible at the coarsest level that there are
  // no windows and thus no nestsets. We need to add
  // an all zero field so that ghost field is consistent
  // across all domains
  if(!dom.has_path("nestsets/"+nestset_name))
  {
    // we alrady init'd it to zero if its a new field, so we are good to go
    // if it wasnt' a new field, it should already have the correct values
    return;
  }

  // ok, now paint
  conduit::Node &nestset = dom["nestsets/"+nestset_name];

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

std::set<std::string> topology_names(const conduit::Node &dataset)
{
  std::set<std::string> topos;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("topologies"))
    {
      const int num_topos = dom["topologies"].number_of_children();
      std::vector<std::string> topo_names = dom["topologies"].child_names();
      for(int t = 0; t < num_topos; t++)
      {
        topos.insert(topo_names[t]);
      }
    }
  }

  gather_strings(topos);

  return topos;
}

bool field_is_float32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float32();
  }
}

bool field_is_float64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float64();
  }
}

bool field_is_int32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int32();
  }
}

bool field_is_int64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int64();
  }
}

Array<double>
centroids(const conduit::Node &domain, const std::string topo)
{
  const conduit::Node &n_topo = domain["topologies"][topo];
  const std::string mesh_type = n_topo["type"].as_string();
  const std::string coords_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = domain["coordsets/"+coords_name];
  Array<double> res;

  detail::CentroidFunctor func;
  exec_dispatch_mesh(n_coords,n_topo, func);
  return res;
}

Array<double>
vertices(const conduit::Node &domain, const std::string topo)
{
  const conduit::Node &n_topo = domain["topologies"][topo];
  const std::string mesh_type = n_topo["type"].as_string();
  const std::string coords_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = domain["coordsets/"+coords_name];

  detail::VertexFunctor func;
  exec_dispatch_mesh(n_coords,n_topo, func);
  return func.m_vertices;
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
