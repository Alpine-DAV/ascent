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

#include <limits>

#include <flow_workspace.hpp>

#ifdef ASCENT_MPI_ENABLED
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

bool at_least_one(bool local)
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
  int m_dims[3] = {0,0,0};
  bool m_is_2d = true;

  UniformCoords(const conduit::Node &n_coords)
  {
    populate(n_coords);
  }

  void populate(const conduit::Node &n_coords)
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
get_num_indices(const std::string shape_type)
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
    ASCENT_ERROR("Unsupported cell type "<<shape_type);
  }
  return num;
}

void logical_index_2d(int *idx, const int point_index, const int *dims)
{
  idx[0] = point_index % dims[0];
  idx[1] = point_index / dims[0];
}

void logical_index_3d(int *idx, const int point_index, const int *dims)
{
  idx[0] = point_index % dims[0];
  idx[1] = (point_index / dims[0]) % dims[1];
  idx[2] = point_index / (dims[0] * dims[1]);
}

void get_cell_indices(const conduit::Node &n_topo,
                      const int index,
                      std::vector<int> &indices)
{

  const std::string mesh_type = n_topo["type"].as_string();
  if(mesh_type == "unstructured")
  {
    // supports only single cell type
    const conduit::Node &n_topo_eles = n_topo["elements"];

    // get the shape
    const std::string ele_shape = n_topo_eles["shape"].as_string();
    const int num_indices = get_num_indices(ele_shape);

    indices.resize(num_indices);
    // look up the connectivity
    const conduit::Node &n_topo_conn = n_topo_eles["connectivity"];
    const conduit::int32_array conn_a = n_topo_conn.value();
    const int offset = index * num_indices;
    for(int i = 0; i < num_indices; ++i)
    {
      indices.push_back(conn_a[offset + i]);
    }
  }
  else
  {
    bool is_2d = true;
    int point_dims[3] = {0, 0, 0};
    point_dims[0] = n_topo["elements/dims/i"].to_int32() + 1;
    point_dims[1] = n_topo["elements/dims/j"].to_int32() + 1;
    n_topo.print();
    if(n_topo.has_path("elements/dims/j"))
    {
      point_dims[2] = n_topo["elements/dims/k"].to_int32() + 1;
      is_2d = false;
    }

    const int cell_dims[3] = {point_dims[0] - 1,
                              point_dims[1] - 1,
                              point_dims[2] - 1};

    int cell_index[3] = {0, 0, 0};
    if(is_2d)
    {
      indices.resize(4);
      logical_index_2d(cell_index, index, cell_dims);

      indices[0] = cell_index[1] * point_dims[0] + cell_index[0];
      indices[1] = indices[0] + 1;
      indices[2] = indices[1] + point_dims[0];
      indices[3] = indices[2] - 1;
    }
    else
    {
      indices.resize(8);
      logical_index_3d(cell_index, index, cell_dims);

      indices[0] = (cell_index[2] * point_dims[1] + cell_index[1]) * point_dims[0] + cell_index[0];
      indices[1] = indices[0] + 1;
      indices[2] = indices[1] + point_dims[1];
      indices[3] = indices[2] - 1;
      indices[4] = indices[0] + point_dims[0] * point_dims[2];
      indices[5] = indices[4] + 1;
      indices[6] = indices[5] + point_dims[1];
      indices[7] = indices[6] - 1;
    }


  }
}


conduit::Node
get_uniform_point(const conduit::Node &n_coords, const int &index)
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

  double point[3];
  point[0] = coords.m_origin[0] + logical_index[0] * coords.m_spacing[0];
  point[1] = coords.m_origin[1] + logical_index[1] * coords.m_spacing[1];
  point[2] = coords.m_origin[2] + logical_index[2] * coords.m_spacing[2];

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
    point[0] = x_a[logical_index[0]];
    point[1] = y_a[logical_index[1]];
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
// ----------------------  cell locations ---------------------------------
conduit::Node
get_uniform_cell(const conduit::Node &n_coords, const int &index)
{

  UniformCoords coords(n_coords);

  int logical_index[3] = {0, 0, 0};
  const int cell_dims[3] = {coords.m_dims[0] - 1,
                            coords.m_dims[1] - 1,
                            coords.m_dims[2] - 1};

  if(coords.m_is_2d)
  {
    logical_index_2d(logical_index, index, cell_dims);
  }
  else
  {
    logical_index_3d(logical_index, index, cell_dims);
  }

  // cell logical index will be the lower left point index

  double point[3] = {0., 0., 0.};
  point[0] = coords.m_origin[0] + logical_index[0] * coords.m_spacing[0] + coords.m_spacing[0] * 0.5;
  point[1] = coords.m_origin[1] + logical_index[1] * coords.m_spacing[1] + coords.m_spacing[1] * 0.5;
  point[2] = coords.m_origin[2] + logical_index[2] * coords.m_spacing[2] + coords.m_spacing[2] * 0.5;

  conduit::Node res;
  res.set(point,3);
  return res;
}

conduit::Node
get_rectilinear_cell(const conduit::Node &n_coords, const int &index)
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
  const int cell_dims[3] = {dims[0] - 1,
                            dims[1] - 1,
                            dims[2] - 1};

  double point[3] = {0., 0., 0.};

  int logical_index[3] = {0, 0, 0};

  if(dims[2] == 0)
  {
    logical_index_2d(logical_index, index, cell_dims);
  }
  else
  {
    logical_index_3d(logical_index, index, cell_dims);
  }

  if(is_float64)
  {
    conduit::float64_array x_a = n_coords["values/x"].value();
    conduit::float64_array y_a = n_coords["values/y"].value();
    point[0] = (x_a[logical_index[0]] + x_a[logical_index[0] + 1]) * 0.5;
    point[1] = (y_a[logical_index[1]] + y_a[logical_index[1] + 1]) * 0.5;
    if(dims[2] != 0)
    {
      conduit::float64_array z_a = n_coords["values/z"].value();
      point[2] = (z_a[logical_index[2]] + z_a[logical_index[2] + 1]) * 0.5;
    }
  }
  else
  {
    conduit::float32_array x_a = n_coords["values/x"].value();
    conduit::float32_array y_a = n_coords["values/y"].value();
    point[0] = (x_a[logical_index[0]] + x_a[logical_index[0] + 1]) * 0.5;
    point[1] = (y_a[logical_index[1]] + y_a[logical_index[1] + 1]) * 0.5;
    if(dims[2] != 0)
    {
      conduit::float32_array z_a = n_coords["values/z"].value();
      point[2] = (z_a[logical_index[2]] + z_a[logical_index[2] + 1]) * 0.5;
    }

  }

  conduit::Node res;
  res.set(point,3);
  return res;
}

conduit::Node
get_explicit_cell(const conduit::Node &n_coords,
                  const conduit::Node &n_topo,
                  const int &index)
{
  std::vector<int> conn;
  get_cell_indices(n_topo, index, conn);
  const int num_indices = conn.size();
  double point[3] = {0., 0., 0.};
  for(int i = 0; i < num_indices; ++i)
  {
    int point_index = conn[i];
    conduit::Node n_point = get_explicit_point(n_coords, point_index);
    double * ptr = n_point.value();
    point[0] += ptr[0];
    point[1] += ptr[1];
    point[2] += ptr[2];
  }

  point[0] /= double(num_indices);
  point[1] /= double(num_indices);
  point[2] /= double(num_indices);

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

conduit::Node
cell_location(const conduit::Node &domain,
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
    res = detail::get_uniform_cell(n_coords, index);
  }
  else if(mesh_type == "rectilinear")
  {
    res = detail::get_rectilinear_cell(n_coords, index);
  }
  else if(mesh_type == "unstructured" || mesh_type == "structured")
  {
    res = detail::get_explicit_cell(n_coords, n_topo, index);
  }
  else
  {
    ASCENT_ERROR("The Architect: unknown mesh type: '"<<mesh_type<<"'");
  }

  return res;
}

bool is_scalar_field(const conduit::Node &dataset, const std::string &field_name)
{
  bool is_scalar = false;
  bool has_field = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_field && dom.has_path("fields/"+field_name))
    {
      has_field = true;
      const conduit::Node &n_field = dom["fields/"+field_name];
      const int num_children = n_field["values"].number_of_children();
      if(num_children == 0)
      {
        is_scalar = true;
      }
    }
  }
  return is_scalar;
}

bool has_field(const conduit::Node &dataset, const std::string &field_name)
{
  bool has_field = false;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_field && dom.has_path("fields/"+field_name))
    {
      has_field = true;
    }
  }
  // check to see if the field exists in any rank
  has_field = detail::at_least_one(has_field);
  return has_field;
}

conduit::Node
field_min(const conduit::Node &dataset,
          const std::string &field)
{
  double min_value = std::numeric_limits<double>::max();

  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/"+field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_min(dom[path]);
      res.print();
      double a_min = res["value"].to_float64();
      std::cout<<"min "<<min_value<<"  current "<<a_min<<"\n";
      if(a_min < min_value)
      {
        min_value = a_min;
        index = res["index"].as_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
      }
    }
  }

  std::cout<<"max value "<<min_value<<"\n";
  std::cout<<"index "<<index<<"\n";

  const std::string assoc_str = dataset.child(0)["fields/"
                                + field + "/association"].as_string();

  conduit::Node loc;
  if(assoc_str == "vertex")
  {
    loc = point_location(dataset.child(domain),index);
  }
  else if(assoc_str == "element")
  {
    loc = cell_location(dataset.child(domain),index);
  }
  else
  {
    ASCENT_ERROR("Location for "<<assoc_str<<" not implemented");
  }

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
  MPI_Allreduce( &minloc, &minloc_res, 1, MPI_DOUBLE_INT, MPI_MINLOC, mpi_comm);
  min_value = minloc.value;

  double * ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, minloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, minloc_res.rank, mpi_comm);

  loc.set(ploc, 3);
  if(rank == 0)
  {
    std::cout<<"Winning rank = "<<minloc_res.rank<<"\n";
    std::cout<<"loc ";
    loc.print();
  }
  rank = minloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["position"] = loc;
  res["value"] = min_value;

  return res;
}

conduit::Node
field_avg(const conduit::Node &dataset,
          const std::string &field)
{
  double sum = 0.;
  long long int count = 0;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/"+field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_sum(dom[path]);
      res.print();
      double a_sum = res["value"].to_float64();
      long long int a_count = res["count"].to_int64();
      std::cout<<"sum "<<sum<<" current sum "<<a_sum<<"\n";
      sum += a_sum;
      count += a_count;
    }
  }

  std::cout<<"sum "<<sum<<"\n";

#ifdef ASCENT_MPI_ENABLED
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);
  double global_sum;
  MPI_Allreduce( &sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

  long long int global_count;
  MPI_Allreduce( &count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, mpi_comm);

  sum = global_sum;
  count = global_count;
#endif
  double avg = sum / double(count);
  conduit::Node res;
  res["value"] = avg;

  return res;
}

conduit::Node
field_max(const conduit::Node &dataset,
          const std::string &field)
{
  double max_value = std::numeric_limits<double>::lowest();

  int domain = -1;
  int domain_id = -1;
  int index = -1;

  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/"+field))
    {
      const std::string path = "fields/" + field + "/values";
      conduit::Node res;
      res = array_max(dom[path]);
      res.print();
      double a_max = res["value"].to_float64();
      std::cout<<"max "<<max_value<<"  current "<<a_max<<"\n";
      if(a_max > max_value)
      {
        max_value = a_max;
        index = res["index"].as_int32();
        domain = i;
        domain_id = dom["state/domain_id"].to_int32();
      }
    }
  }

  std::cout<<"max value "<<max_value<<"\n";
  std::cout<<"index "<<index<<"\n";

  const std::string assoc_str = dataset.child(0)["fields/"
                                + field + "/association"].as_string();

  conduit::Node loc;
  if(assoc_str == "vertex")
  {
    loc = point_location(dataset.child(domain),index);
  }
  else if(assoc_str == "element")
  {
    loc = cell_location(dataset.child(domain),index);
  }
  else
  {
    ASCENT_ERROR("Location for "<<assoc_str<<" not implemented");
  }

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
  MPI_Allreduce( &maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);
  max_value = maxloc.value;

  double * ploc = loc.as_float64_ptr();
  MPI_Bcast(ploc, 3, MPI_DOUBLE, maxloc_res.rank, mpi_comm);
  MPI_Bcast(&domain_id, 1, MPI_INT, maxloc_res.rank, mpi_comm);

  loc.set(ploc, 3);
  if(rank == 0)
  {
    std::cout<<"Winning rank = "<<maxloc_res.rank<<"\n";
    std::cout<<"loc ";
    loc.print();
  }
  rank = maxloc_res.rank;
#endif
  res["rank"] = rank;
  res["domain_id"] = domain_id;
  res["position"] = loc;
  res["value"] = max_value;

  return res;
}

conduit::Node
get_state_var(const conduit::Node &dataset,
              const std::string &var_name)
{
  bool has_state = false;
  conduit::Node state;
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(!has_state && dom.has_path("state/"+var_name))
    {
      has_state = true;
      state = dom["state/"+var_name];
    }
  }

  // TODO: make sure everyone has this
  if(!has_state)
  {
    ASCENT_ERROR("Unable to retrieve state variable '"<<var_name<<"'");
  }
  return state;
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





