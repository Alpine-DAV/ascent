// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/blueprint_low_order.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/error.hpp>
#include <dray/array_utils.hpp>
#include "conduit_blueprint.hpp"

using namespace conduit;

namespace dray
{

namespace detail
{

// vtk to lexagraphical ordering (zyx)
constexpr int32 hex_conn_map[8] = {0, 1, 4, 5, 3, 2, 7, 6 };
constexpr int32 quad_conn_map[4] = {0, 1, 3, 2};
constexpr int32 tet_conn_map[4] = {0, 1, 2, 3};
constexpr int32 tri_conn_map[3] = {0, 1, 2};

int32 dofs_per_elem(const std::string shape)
{
  int32 dofs = 0;
  if(shape == "tri")
  {
    dofs = 3;
  }
  else if(shape == "tet" || shape == "quad")
  {
    dofs = 4;
  }
  else if(shape == "hex")
  {
    dofs = 8;
  }
  return dofs;
}

template<typename T>
Array<int32>
convert_conn(const conduit::Node &n_conn,
             const std::string shape,
             int32 &num_elems)
{
  const int conn_size = n_conn.dtype().number_of_elements();
  Array<int32> conn;
  conn.resize(conn_size);
  int32 *conn_ptr = conn.get_host_ptr();

  const int num_dofs = dofs_per_elem(shape);
  num_elems = conn_size / num_dofs;

  conduit::DataArray<int32> conn_array = n_conn.value();

  const int32 *map = shape == "hex" ? hex_conn_map :
                     shape == "quad" ? quad_conn_map :
                     shape == "tri" ? tri_conn_map : tet_conn_map;
  for(int32 i = 0; i < num_elems; ++i)
  {
    const int32 offset = i * num_dofs;
    for(int32 dof = 0; dof < num_dofs; ++dof)
    {
      conn_ptr[offset + dof] = conn_array[offset + map[dof]];
    }
  }

  return conn;
}

Array<Vec<Float,1>>
copy_conduit_scalar_array(const conduit::Node &n_vals)
{
  int num_vals = n_vals.dtype().number_of_elements();
  Array<Vec<Float,1>> values;
  values.resize(num_vals);

  Vec<Float,1> *values_ptr = values.get_host_ptr();

  if(n_vals.dtype().is_float32())
  {
    const float *n_values_ptr = n_vals.value();
    for(int32 i = 0; i < num_vals; ++i)
    {
      values_ptr[i][0] = n_values_ptr[i];
    }
  }
  else if(n_vals.dtype().is_float64())
  {
    const double *n_values_ptr = n_vals.value();
    for(int32 i = 0; i < num_vals; ++i)
    {
      values_ptr[i][0] = n_values_ptr[i];
    }
  }
  else
  {
    DRAY_ERROR("Unsupported copy type");
  }
  return values;
}

Array<Vec<Float,2>>
copy_conduit_mcarray_2d(const conduit::Node &n_vals)
{

#ifdef DRAY_DOUBLE_PRECISION
  float64_accessor comp_0_vals = n_vals[0].value();
  float64_accessor comp_1_vals = n_vals[1].value();
#else
  float32_accessor comp_0_vals = n_vals[0].value();
  float32_accessor comp_1_vals = n_vals[1].value();
#endif

  int num_vals = comp_0_vals.number_of_elements();
  Array<Vec<Float,2>> values;
  values.resize(num_vals);

  Vec<Float,2> *values_ptr = values.get_host_ptr();

  for(int32 i = 0; i < num_vals; ++i)
  {
      values_ptr[i][0] = comp_0_vals[i];
      values_ptr[i][1] = comp_1_vals[i];
  }

  return values;
}

Array<Vec<Float,3>>
copy_conduit_mcarray_3d(const conduit::Node &n_vals)
{

#ifdef DRAY_DOUBLE_PRECISION
  float64_accessor comp_0_vals = n_vals[0].value();
  float64_accessor comp_1_vals = n_vals[1].value();
  float64_accessor comp_2_vals = n_vals[2].value();
#else
  float32_accessor comp_0_vals = n_vals[0].value();
  float32_accessor comp_1_vals = n_vals[1].value();
  float32_accessor comp_2_vals = n_vals[2].value();
#endif

  int num_vals = comp_0_vals.number_of_elements();
  Array<Vec<Float,3>> values;
  values.resize(num_vals);

  Vec<Float,3> *values_ptr = values.get_host_ptr();

  for(int32 i = 0; i < num_vals; ++i)
  {
      values_ptr[i][0] = comp_0_vals[i];
      values_ptr[i][1] = comp_1_vals[i];
      values_ptr[i][2] = comp_2_vals[i];
  }

  return values;
}


Array<Vec<Float,3>>
import_explicit_coords(const conduit::Node &n_coords)
{
    int32 nverts = n_coords["values/x"].dtype().number_of_elements();

    Array<Vec<Float,3>> coords;
    coords.resize(nverts);
    Vec<Float,3> *coords_ptr = coords.get_host_ptr();

    int32 ndims = 2;
    if(n_coords["values"].has_path("z"))
    {
      ndims = 3;
    }

    bool is_float = n_coords["values/x"].dtype().is_float32();

    if(is_float)
    {
      conduit::float32_array x_array = n_coords["values/x"].value();
      conduit::float32_array y_array = n_coords["values/y"].value();
      conduit::float32_array z_array;

      if(ndims == 3)
      {
        z_array = n_coords["values/z"].value();
      }

      for(int32 i = 0; i < nverts; ++i)
      {
        Vec<Float,3> point;
        point[0] = x_array[i];
        point[1] = y_array[i];
        point[2] = 0.f;
        if(ndims == 3)
        {
          point[2] = z_array[i];
        }
        coords_ptr[i] = point;
      }
    }
    else
    {
      conduit::float64_array x_array = n_coords["values/x"].value();
      conduit::float64_array y_array = n_coords["values/y"].value();
      conduit::float64_array z_array;

      if(ndims == 3)
      {
        z_array = n_coords["values/z"].value();
      }

      for(int32 i = 0; i < nverts; ++i)
      {
        Vec<Float,3> point;
        point[0] = x_array[i];
        point[1] = y_array[i];
        point[2] = 0.f;
        if(ndims == 3)
        {
          point[2] = z_array[i];
        }

        coords_ptr[i] = point;
      }
    }

    return coords;
}


void
logical_index_2d(Vec<int32,3> &idx,
                 const int32 index,
                 const Vec<int32,3> &dims)
{
  idx[0] = index % dims[0];
  idx[1] = index / dims[0];
}

void
logical_index_3d(Vec<int32,3> &idx,
                 const int32 index,
                 const Vec<int32,3> &dims)
{
  idx[0] = index % dims[0];
  idx[1] = (index / dims[0]) % dims[1];
  idx[2] = index / (dims[0] * dims[1]);
}

Array<int32>
structured_conn(const Vec<int32,3> point_dims,
                bool is_3d,
                int32 &n_elems)
{
  Vec<int32,3> cell_dims;
  cell_dims[0] = point_dims[0] - 1;
  cell_dims[1] = point_dims[1] - 1;
  n_elems = cell_dims[0] * cell_dims[1];
  int32 n_verts = point_dims[0] * point_dims[1];
  if(is_3d)
  {
    cell_dims[2] = point_dims[2] - 1;
    n_verts *= point_dims[2];
    n_elems *= cell_dims[2];
  }

  const int32 verts_per_elem = is_3d ? 8 : 4;

  Array<int32> conn;
  conn.resize(n_elems * verts_per_elem);
  int32 *conn_ptr = conn.get_host_ptr();

  for(int32 i = 0; i < n_elems; ++i)
  {
    const int32 offset = i * verts_per_elem;
    Vec<int32,3> idx;

    if(!is_3d)
    {
      detail::logical_index_2d(idx, i, cell_dims);
      // this is the vtk version
      //conn_ptr[offset + 0] = idx[1] * dims[0] + idx[0];
      //conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      //conn_ptr[offset + 2] = conn_ptr[offset + 1] + dims[0];
      //conn_ptr[offset + 3] = conn_ptr[offset + 2] - 1;
      // this is the dray version (lexagraphical ordering x,y,z)
      conn_ptr[offset + 0] = idx[1] * point_dims[0] + idx[0];
      conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      conn_ptr[offset + 2] = conn_ptr[offset + 0] + point_dims[0];
      conn_ptr[offset + 3] = conn_ptr[offset + 2] + 1;
    }
    else
    {
      detail::logical_index_3d(idx, i, cell_dims);
      // this is the vtk version
      //conn_ptr[offset + 0] = (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0];
      //conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      //conn_ptr[offset + 2] = conn_ptr[offset + 1] + dims[1];
      //conn_ptr[offset + 3] = conn_ptr[offset + 2] - 1;
      //conn_ptr[offset + 4] = conn_ptr[offset + 0] + dims[0] * dims[2];
      //conn_ptr[offset + 5] = conn_ptr[offset + 4] + 1;
      //conn_ptr[offset + 6] = conn_ptr[offset + 5] + dims[1];
      //conn_ptr[offset + 7] = conn_ptr[offset + 6] - 1;
      // this is the dray version (lexagraphical ordering x,y,z)
      conn_ptr[offset + 0] = (idx[2] * point_dims[1] + idx[1]) * point_dims[0] + idx[0];
      conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;

      // advance in y
      conn_ptr[offset + 2] = conn_ptr[offset + 0] + point_dims[0];
      conn_ptr[offset + 3] = conn_ptr[offset + 2] + 1;

      // advance in z
      conn_ptr[offset + 4] = conn_ptr[offset + 0] + point_dims[0] * point_dims[1];
      conn_ptr[offset + 5] = conn_ptr[offset + 4] + 1;
      // advance in y
      conn_ptr[offset + 6] = conn_ptr[offset + 4] + point_dims[0];
      conn_ptr[offset + 7] = conn_ptr[offset + 6] + 1;
    }
  }
  return conn;
}

void
import_scalar_field(const Node &n_field,
                    int num_elems,
                    int order,
                    const std::string &assoc,
                    const std::string &shape,
                    const std::string &topo,
                    Array<int32> &ctrl_idx,
                    DataSet &dataset)
{
    // if we are elemen assoced (order == 0), we have 1 dof per element
    int32 num_dofs_per_elem = 1;

    // if we are vertex assoced (order == 1), # of dofs depends on shape
    if( assoc == "vertex" )
    {
      // todo: this will depend on shape type
      num_dofs_per_elem = detail::dofs_per_elem(shape);
    }

    const conduit::Node &n_vals = n_field["values"].number_of_children() == 0
         ? n_field["values"] : n_field["values"].child(0);

    // copy conduit array into dray array
    Array<Vec<Float,1>> values = detail::copy_conduit_scalar_array(n_vals);

    const std::string field_name = n_field.name();

    // create the base grid func
    GridFunction<1> gf;
    gf.m_values    = values;
    gf.m_ctrl_idx  = ctrl_idx;
    gf.m_el_dofs   = num_dofs_per_elem;
    gf.m_size_el   = num_elems;
    gf.m_size_ctrl = ctrl_idx.size();

    std::shared_ptr<Field> field;

    if(shape == "quad")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<QuadScalar_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<QuadScalar_P0>>(gf, order, field_name);
        }
    }
    else if(shape == "hex")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<HexScalar_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<HexScalar_P0>>(gf, order, field_name);
        }
    }
    else if(shape == "tri")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<TriScalar_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<TriScalar_P0>>(gf, order, field_name);
        }
    }
    else if(shape == "tet")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<TetScalar_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<TetScalar_P0>>(gf, order, field_name);
        }
    }
    else
    {
        DRAY_ERROR("Unsupported field shape '"<<shape<<"' assoc '"<<assoc<<"'");
    }

    field->mesh_name(topo);
    dataset.add_field(field);
}


void
import_vector_field_2d(const Node &n_field,
                       int num_elems,
                       int order,
                       const std::string &assoc,
                       const std::string &shape,
                       const std::string &topo,
                       Array<int32> &ctrl_idx,
                       DataSet &dataset)
{
    // if we are elemen assoced (order == 0), we have 1 dof per element
    int32 num_dofs_per_elem = 1;

    // if we are vertex assoced (order == 1), # of dofs depends on shape
    if( assoc == "vertex" )
    {
      // todo: this will depend on shape type
      num_dofs_per_elem = detail::dofs_per_elem(shape);
    }


    // copy conduit array into dray array
    Array<Vec<Float,2>> values = detail::copy_conduit_mcarray_2d(n_field["values"]);

    const std::string field_name = n_field.name();

    // create the base grid func
    GridFunction<2> gf;
    gf.m_values    = values;
    gf.m_ctrl_idx  = ctrl_idx;
    gf.m_el_dofs   = num_dofs_per_elem;
    gf.m_size_el   = num_elems;
    gf.m_size_ctrl = ctrl_idx.size();

    std::shared_ptr<Field> field;

    if(shape == "quad")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<QuadVector_2D_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<QuadVector_2D_P0>>(gf, order, field_name);
        }
    }
    // TODO 2D vector fields for 3D meshes?
    // else if(shape == "hex")
    // {
    //     if(assoc == "vertex")
    //     {
    //         field = std::make_shared<UnstructuredField<HexScalar_P1>>(gf, order, field_name);
    //     }
    //     else
    //     {
    //         field = std::make_shared<UnstructuredField<HexScalar_P0>>(gf, order, field_name);
    //     }
    // }
    else if(shape == "tri")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<TriVector_2D_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<TriVector_2D_P0>>(gf, order, field_name);
        }
    }
    // TODO 2D vector fields for 3D meshes?
    // else if(shape == "tet")
    // {
    //     if(assoc == "vertex")
    //     {
    //         field = std::make_shared<UnstructuredField<TetScalar_P1>>(gf, order, field_name);
    //     }
    //     else
    //     {
    //         field = std::make_shared<UnstructuredField<TetScalar_P0>>(gf, order, field_name);
    //     }
    // }
    else
    {
        DRAY_ERROR("Unsupported field shape '"<<shape<<"' assoc '"<<assoc<<"'");
    }

    field->mesh_name(topo);
    dataset.add_field(field);
}

void
import_vector_field_3d(const Node &n_field,
                       int num_elems,
                       int order,
                       const std::string &assoc,
                       const std::string &shape,
                       const std::string &topo,
                       Array<int32> &ctrl_idx,
                       DataSet &dataset)
{
    // if we are elemen assoced (order == 0), we have 1 dof per element
    int32 num_dofs_per_elem = 1;

    // if we are vertex assoced (order == 1), # of dofs depends on shape
    if( assoc == "vertex" )
    {
      // todo: this will depend on shape type
      num_dofs_per_elem = detail::dofs_per_elem(shape);
    }


    // copy conduit array into dray array
    Array<Vec<Float,3>> values = detail::copy_conduit_mcarray_3d(n_field["values"]);

    const std::string field_name = n_field.name();

    // create the base grid func
    GridFunction<3> gf;
    gf.m_values    = values;
    gf.m_ctrl_idx  = ctrl_idx;
    gf.m_el_dofs   = num_dofs_per_elem;
    gf.m_size_el   = num_elems;
    gf.m_size_ctrl = ctrl_idx.size();

    std::shared_ptr<Field> field;

    // TODO 3D vector fields for 2D meshes?
    // if(shape == "quad")
    // {
    //     if(assoc == "vertex")
    //     {
    //         field = std::make_shared<UnstructuredField<QuadVector_2D_P1>>(gf, order, field_name);
    //     }
    //     else
    //     {
    //         field = std::make_shared<UnstructuredField<QuadVector_2D_P0>>(gf, order, field_name);
    //     }
    // }

    // else if(shape == "hex")
    
    if(shape == "hex")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<HexVector_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<HexVector_P0>>(gf, order, field_name);
        }
    }
    // TODO 3D vector fields for 2D meshes?
    // else if(shape == "tri")
    // {
    //     if(assoc == "vertex")
    //     {
    //         field = std::make_shared<UnstructuredField<TriVector_2D_P1>>(gf, order, field_name);
    //     }
    //     else
    //     {
    //         field = std::make_shared<UnstructuredField<TriVector_2D_P0>>(gf, order, field_name);
    //     }
    // }
    else if(shape == "tet")
    {
        if(assoc == "vertex")
        {
            field = std::make_shared<UnstructuredField<TetVector_P1>>(gf, order, field_name);
        }
        else
        {
            field = std::make_shared<UnstructuredField<TetVector_P0>>(gf, order, field_name);
        }
    }
    else
    {
        DRAY_ERROR("Unsupported field shape '"<<shape<<"' assoc '"<<assoc<<"'");
    }

    field->mesh_name(topo);
    dataset.add_field(field);
}


} // namespace detail

DataSet
BlueprintLowOrder::import(const conduit::Node &n_dataset)
{
  DataSet dataset;

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh",n_dataset, info))
  {
    DRAY_ERROR("Import failed to verify "<<info.to_yaml());
  }

  std::map<std::string, std::string> topologies_shapes;
  std::map<std::string, Array<int32>>  topologies_conn;
  const int32 num_topos = n_dataset["topologies"].number_of_children();
  for(int32 i = 0; i < num_topos; ++i)
  {
    const conduit::Node &n_topo = n_dataset["topologies"].child(0);
    const std::string topo_name = n_dataset["topologies"].child_names()[i];

    const std::string coords_name = n_topo["coordset"].as_string();
    const std::string mesh_type = n_topo["type"].as_string();

    const conduit::Node &n_coords = n_dataset["coordsets/"+coords_name];

    Array<int32> conn;
    int32 n_elems = 0;
    std::string shape;

    std::shared_ptr<Mesh> topo;

    if(mesh_type == "uniform")
    {
      topo = import_uniform(n_coords, conn, n_elems, shape);
    }
    else if(mesh_type == "unstructured")
    {
      topo = import_explicit(n_coords, n_topo, conn, n_elems, shape);
    }
    else if(mesh_type == "structured")
    {
      topo = import_structured(n_coords, n_topo, conn, n_elems, shape);
    }
    else
    {
      DRAY_ERROR("not implemented "<<mesh_type);
    }
    topo->name(topo_name);
    dataset.add_mesh(topo);
    topologies_shapes[topo_name] = shape;
    topologies_conn[topo_name] = conn;
  }

  const int32 num_fields = n_dataset["fields"].number_of_children();
  std::vector<std::string> field_names = n_dataset["fields"].child_names();

  Array<int32> element_conn;

  for(int32 i = 0; i < num_fields; ++i)
  {
    const conduit::Node &n_field = n_dataset["fields"].child(i);

    // import_field(n_field,shape,dataset);

    std::string field_topo = n_field["topology"].as_string();
    std::string shape = topologies_shapes[field_topo];

    int32 components = n_field["values"].number_of_children();
    bool is_scalar = components == 0 || components == 1;

    std::string assoc = n_field["association"].as_string();
    const int32 n_elems = dataset.mesh(field_topo)->cells();
    Array<int32> conn = topologies_conn[field_topo];

    // if we are vertex assoced, we will use the 
    // vertex ids from the blueprint connectivity as 
    // the control_index (the map from our field vals to the vertices)

    // if we are element assoced, we use a simple counting
    // as the index
    // the control_index (the map from our field vals to the elements)

    // order == 0 for element assoced
    // order == 1 for vertex assoced
    int order = 1;

    if(assoc != "vertex" )
    {
        order = 0;
        // this will be shared across element asscoed fields, but we defer creation until
        // we actually know we have element assoced fields
        if(element_conn.size() == 0)
        {
            element_conn = array_counting(n_elems, 0, 1);
        }
    }

    if( is_scalar )
    {
        detail::import_scalar_field(n_field, // conduit field node
                                    n_elems, // number of elements
                                    order,   // order
                                    assoc,   // assoc
                                    shape,   // shape
                                    field_topo, // topo name
                                    (order == 1) ? conn : element_conn, // ctrl idx
                                    dataset // add to this dataset
                                    );
    }
    else if( components == 2 )
    {
        detail::import_vector_field_2d(n_field, // conduit field node
                                       n_elems, // number of elements
                                       order,   // order
                                       assoc,   // assoc
                                       shape,   // shape
                                       field_topo, // topo name
                                       (order == 1) ? conn : element_conn, // ctrl idx
                                       dataset // add to this dataset
                                       );
    }
    else if( components == 3 )
    {
        detail::import_vector_field_3d(n_field, // conduit field node
                                       n_elems, // number of elements
                                       order,   // order
                                       assoc,   // assoc
                                       shape,   // shape
                                       field_topo, // topo name
                                       (order == 1) ? conn : element_conn, // ctrl idx
                                       dataset // add to this dataset
                                       );
    }
    else
    {
        DRAY_ERROR("fields with "<<components<< " components are not supported");
    }

  }
  return dataset;
}

std::shared_ptr<Mesh>
BlueprintLowOrder::import_structured(const conduit::Node &n_coords,
                                     const conduit::Node &n_topo,
                                     Array<int32> &conn,
                                     int32 &n_elems,
                                     std::string &shape)
{
  const std::string type = n_coords["type"].as_string();
  const std::string t_type = n_topo["type"].as_string();
  if(type != "explicit" && t_type != "structured")
  {
    DRAY_ERROR("bad matt");
  }

  Array<Vec<Float,3>> coords = detail::import_explicit_coords(n_coords);

  bool is_3d = true;
  if(!n_topo.has_path("elements/dims/k"))
  {
    is_3d = false;
  }

  if(is_3d)
  {
    shape = "hex";
  }
  else
  {
    shape = "quad";
  }

  const conduit::Node &n_topo_eles = n_topo["elements"];
  Vec<int32,3> dims = {{0,0,1}};
  dims[0] = n_topo_eles["dims/i"].to_int32() + 1;
  dims[1] = n_topo_eles["dims/j"].to_int32() + 1;
  if(is_3d)
  {
    dims[2] = n_topo_eles["dims/k"].to_int32() + 1;
  }

  conn = detail::structured_conn(dims, is_3d, n_elems);

  const int32 verts_per_elem = is_3d ? 8 : 4;

  GridFunction<3> gf;
  gf.m_ctrl_idx = conn;
  gf.m_values = coords;
  gf.m_el_dofs = verts_per_elem;
  gf.m_size_el = n_elems;
  gf.m_size_ctrl = conn.size();

  using HexMesh = MeshElem<3u, Tensor, Linear>;
  using QuadMesh = MeshElem<2u, Tensor, Linear>;
  int32 order = 1;

  std::shared_ptr<Mesh> res;
  if(is_3d)
  {
    UnstructuredMesh<HexMesh> mesh (gf, order);
    res = std::make_shared<HexMesh_P1>(mesh);
  }
  else
  {
    UnstructuredMesh<QuadMesh> mesh (gf, order);
    res = std::make_shared<QuadMesh_P1>(mesh);
  }

  return res;
}

std::shared_ptr<Mesh>
BlueprintLowOrder::import_explicit(const conduit::Node &n_coords,
                                   const conduit::Node &n_topo,
                                  Array<int32> &conn,
                                  int32 &n_elems,
                                  std::string &shape)
{
  const std::string type = n_coords["type"].as_string();
  if(type != "explicit")
  {
    DRAY_ERROR("bad matt");
  }

  Array<Vec<Float,3>> coords = detail::import_explicit_coords(n_coords);

  const conduit::Node &n_topo_eles = n_topo["elements"];
  std::string ele_shape = n_topo_eles["shape"].as_string();
  shape = ele_shape;
  bool supported_shape = false;

  if(ele_shape == "hex" || ele_shape == "tet" ||
     ele_shape ==  "quad" || ele_shape == "tri")
  {
    supported_shape = true;
  }

  if(!supported_shape)
  {
    DRAY_ERROR("Shape '"<<ele_shape<<"' not currently supported");
  }

  const conduit::Node &n_topo_conn = n_topo_eles["connectivity"];
  n_elems = 0;
  if(n_topo_conn.dtype().is_int32())
  {
    conn = detail::convert_conn<int32>(n_topo_conn, ele_shape, n_elems);
  }
  else if(n_topo_conn.dtype().is_int64())
  {
    conn = detail::convert_conn<int64>(n_topo_conn, ele_shape, n_elems);
  }
  else
  {
    DRAY_ERROR("Unsupported conn data type");
  }

  int32 verts_per_elem = detail::dofs_per_elem(ele_shape);

  GridFunction<3> gf;
  gf.m_ctrl_idx = conn;
  gf.m_values = coords;
  gf.m_el_dofs = verts_per_elem;
  gf.m_size_el = n_elems;
  gf.m_size_ctrl = conn.size();

  using HexMesh = MeshElem<3u, Tensor, Linear>;
  using QuadMesh = MeshElem<2u, Tensor, Linear>;
  using TetMesh = MeshElem<3u, Simplex, Linear>;
  using TriMesh = MeshElem<2u, Simplex, Linear>;
  int32 order = 1;

  std::shared_ptr<Mesh> res;
  if(ele_shape == "tri")
  {
    UnstructuredMesh<TriMesh> mesh (gf, order);
    res = std::make_shared<TriMesh_P1>(mesh);
  }
  else if(ele_shape == "tet")
  {
    UnstructuredMesh<TetMesh> mesh (gf, order);
    res = std::make_shared<TetMesh_P1>(mesh);
  }
  else if(ele_shape == "quad")
  {
    UnstructuredMesh<QuadMesh> mesh (gf, order);
    res = std::make_shared<QuadMesh_P1>(mesh);
  }
  else if(ele_shape == "hex")
  {
    UnstructuredMesh<HexMesh> mesh (gf, order);
    res = std::make_shared<HexMesh_P1>(mesh);
  }

  return res;
}

std::shared_ptr<Mesh>
BlueprintLowOrder::import_uniform(const conduit::Node &n_coords,
                                  Array<int32> &conn,
                                  int32 &n_elems,
                                  std::string &shape)
{

  const std::string type = n_coords["type"].as_string();
  if(type != "uniform")
  {
    DRAY_ERROR("bad matt");
  }

  const conduit::Node &n_dims = n_coords["dims"];

  Vec<int32,3> dims;
  dims[0] = n_dims["i"].to_int();
  dims[1] = n_dims["j"].to_int();
  dims[2] = 1;

  bool is_2d = true;
  if(n_dims.has_path("k"))
  {
    is_2d = false;
    dims[2] = n_dims["k"].to_int();
  }
  if(is_2d)
  {
    shape = "quad";
  }
  else
  {
    shape = "hex";
  }

  float64 origin_x = 0.0;
  float64 origin_y = 0.0;
  float64 origin_z = 0.0;

  float64 spacing_x = 1.0;
  float64 spacing_y = 1.0;
  float64 spacing_z = 1.0;

  if(n_coords.has_child("origin"))
  {
    const conduit::Node &n_origin = n_coords["origin"];

    if(n_origin.has_child("x"))
    {
      origin_x = n_origin["x"].to_float64();
    }

    if(n_origin.has_child("y"))
    {
      origin_y = n_origin["y"].to_float64();
    }

    if(n_origin.has_child("z"))
    {
      origin_z = n_origin["z"].to_float64();
    }
  }

  if(n_coords.has_path("spacing"))
  {
    const conduit::Node &n_spacing = n_coords["spacing"];

    if(n_spacing.has_path("dx"))
    {
      spacing_x = n_spacing["dx"].to_float64();
    }

    if(n_spacing.has_path("dy"))
    {
      spacing_y = n_spacing["dy"].to_float64();
    }

    if(n_spacing.has_path("dz"))
    {
      spacing_z = n_spacing["dz"].to_float64();
    }
  }

  Array<Vec<Float,3>> coords;
  const int32 n_verts = dims[0] * dims[1] * dims[2];
  coords.resize(n_verts);
  Vec<Float,3> *coords_ptr = coords.get_host_ptr();

  for(int32 i = 0; i < n_verts; ++i)
  {
    Vec<int32,3> idx;
    if(is_2d)
    {
      detail::logical_index_2d(idx, i, dims);
    }
    else
    {
      detail::logical_index_3d(idx, i, dims);
    }

    Vec<Float,3> point;
    point[0] = origin_x + idx[0] * spacing_x;
    point[1] = origin_y + idx[1] * spacing_y;
    if(is_2d)
    {
      point[2] = 0.f;
    }
    else
    {
      point[2] = origin_z + idx[2] * spacing_z;
    }

    coords_ptr[i] = point;
  }

  conn = detail::structured_conn(dims, !is_2d, n_elems);
  const int32 verts_per_elem = is_2d ? 4 : 8;

  GridFunction<3> gf;
  gf.m_ctrl_idx = conn;
  gf.m_values = coords;
  gf.m_el_dofs = verts_per_elem;
  gf.m_size_el = n_elems;
  gf.m_size_ctrl = conn.size();

  using HexMesh = MeshElem<3u, Tensor, Linear>;
  using QuadMesh = MeshElem<2u, Tensor, Linear>;
  int32 order = 1;

  std::shared_ptr<Mesh> res;
  if(is_2d)
  {
    UnstructuredMesh<QuadMesh> mesh (gf, order);
    res = std::make_shared<QuadMesh_P1>(mesh);
  }
  else
  {
    UnstructuredMesh<HexMesh> mesh (gf, order);
    res = std::make_shared<HexMesh_P1>(mesh);
  }

  return res;
}

// Rearrange dray Conduit node to one that conforms to Blueprint.
void
BlueprintLowOrder::to_blueprint(const conduit::Node &dray_rep, conduit::Node &n)
{
    //dray_rep.print();

    const conduit::Node &meshes = dray_rep["meshes"];
    const conduit::Node &mesh = meshes[0];
    std::string type_name(mesh["type_name"].as_string());
    bool _2D = type_name.find("2D") == 0;

    // Add the coordinates.
    const conduit::Node &mgf = mesh["grid_function"];
    int npts = mgf["values_size"].to_int();
    const conduit::Node &values = mgf["values"];
    auto coords_ptr = reinterpret_cast<Float *>(const_cast<void*>(values.data_ptr()));
    n["coordsets/coords/type"] = "explicit";
    n["coordsets/coords/values/x"].set_external(coords_ptr, npts, 0, 3 * sizeof(Float));
    n["coordsets/coords/values/y"].set_external(coords_ptr, npts, sizeof(Float), 3 * sizeof(Float));
    n["coordsets/coords/values/z"].set_external(coords_ptr, npts, 2*sizeof(Float), 3 * sizeof(Float));

    // Add the topology
    int dofs_per_element = mgf["dofs_per_element"].to_int();
    if(dofs_per_element == 8)
    {
        conduit::Node &n_topo = n["topologies/topology"];
        n_topo["coordset"] = "coords";
        n_topo["type"] = "unstructured";
        n_topo["elements/shape"] = "hex";
#if 1
        // node reordering seems to be needed.
        int conn_size = mgf["conn_size"].to_int();
        int nelem = mgf["num_elements"].to_int();
        auto conn_ptr = reinterpret_cast<int *>(const_cast<void*>(mgf["conn"].data_ptr()));
        n_topo["elements/connectivity"].set(conduit::DataType::int32(nelem * 8));
        auto newconn_ptr = reinterpret_cast<int *>(const_cast<void*>(n_topo["elements/connectivity"].data_ptr()));
        const int reorder[] = {0,1,3,2,4,5,7,6};
        for(int i = 0; i < nelem; i++)
        {
            int *cell_src = conn_ptr + 8 * i;
            int *cell_dest = newconn_ptr + 8 * i;

            //std::cout << "cell " << i << " = {";
            //for(int j = 0; j < 8; j++)
            //    std::cout << cell_src[j] << ", ";
            //std::cout << "}" << std::endl;

            for(int j = 0; j < 8; j++)
                cell_dest[j] = cell_src[reorder[j]];
        }
#else
        n_topo["elements/connectivity"].set_external_node(mgf["conn"]);
#endif
    }
    else if(dofs_per_element == 4)
    {
        conduit::Node &n_topo = n["topologies/topology"];
        n_topo["coordset"] = "coords";
        n_topo["type"] = "unstructured";
        if(_2D)
        {
            n_topo["elements/shape"] = "quad";
            // node reordering seems to be needed.
            int conn_size = mgf["conn_size"].to_int();
            int nelem = mgf["num_elements"].to_int();
            auto conn_ptr = reinterpret_cast<int *>(const_cast<void*>(mgf["conn"].data_ptr()));
            n_topo["elements/connectivity"].set(conduit::DataType::int32(nelem * 4));
            auto newconn_ptr = reinterpret_cast<int *>(const_cast<void*>(n_topo["elements/connectivity"].data_ptr()));
            const int reorder[] = {0,1,3,2};
            for(int i = 0; i < nelem; i++)
            {
                int *cell_src = conn_ptr + 4 * i;
                int *cell_dest = newconn_ptr + 4 * i;

                //std::cout << "cell " << i << " = {";
                //for(int j = 0; j < 4; j++)
                //    std::cout << cell_src[j] << ", ";
                //std::cout << "}" << std::endl;

                for(int j = 0; j < 4; j++)
                    cell_dest[j] = cell_src[reorder[j]];
            }
        }
        else
        {
            n_topo["elements/shape"] = "tet";
            n_topo["elements/connectivity"].set_external_node(mgf["conn"]);
        }
    }
    else if(dofs_per_element == 3)
    {
        conduit::Node &n_topo = n["topologies/topology"];
        n_topo["coordset"] = "coords";
        n_topo["type"] = "unstructured";
        n_topo["elements/shape"] = "tri";
        n_topo["elements/connectivity"].set_external_node(mgf["conn"]);
    }

    // Do fields.
    const conduit::Node &n_fields = dray_rep["fields"];
    conduit::Node &n_outfields = n["fields"];
    for(conduit::index_t i = 0; i < n_fields.number_of_children(); i++)
    {
        const conduit::Node &n_field = n_fields[i];
        const conduit::Node &n_gf = n_field["grid_function"];
        dofs_per_element = n_gf["dofs_per_element"].to_int();
        int phys_dim = n_gf["phys_dim"].to_int();

        conduit::Node &n_outfield = n_outfields[n_field.name()];
        n_outfield["topology"] = "topology";
        int nvalues = n_gf["values_size"].to_int();
        if(dofs_per_element == 1)
        {
            n_outfield["association"] = "element";
        }
        else
        {
            n_outfield["association"] = "vertex";
        }
        if(phys_dim == 1)
        {
            n_outfield["values"].set_external_node(n_gf["values"]);
        }
        else if(phys_dim == 2)
        {
            auto ptr = reinterpret_cast<Float *>(const_cast<void*>(n_gf["values"].data_ptr()));
            n_outfield["values/x"].set_external(ptr, nvalues, 0, 2 * sizeof(Float));
            n_outfield["values/y"].set_external(ptr, nvalues, sizeof(Float), 2 * sizeof(Float));
        }
        else if(phys_dim == 3)
        {
            auto ptr = reinterpret_cast<Float *>(const_cast<void*>(n_gf["values"].data_ptr()));
            n_outfield["values/x"].set_external(ptr, nvalues, 0, 3 * sizeof(Float));
            n_outfield["values/y"].set_external(ptr, nvalues, sizeof(Float), 3 * sizeof(Float));
            n_outfield["values/z"].set_external(ptr, nvalues, 2*sizeof(Float), 3 * sizeof(Float));
        }
    }

    //n.print();
}

} // namespace dray
