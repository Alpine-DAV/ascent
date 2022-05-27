// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray_node_to_dataset.hpp>
#include <dray/error.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>

#include <string>
#include <vector>

namespace dray
{

namespace detail
{

std::vector<std::string> split (std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos)
  {
    token = s.substr (pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back (token);
  }

  res.push_back (s.substr (pos_start));
  return res;
}

template <int32 PhysDim>
GridFunction<PhysDim>
import_grid_function(const conduit::Node &n_gf,
                     int compoments)
{
  GridFunction<PhysDim> gf;
  if(!n_gf.has_path("values"))
  {
    DRAY_ERROR("Grid function missing values");
  }
  if(!n_gf.has_path("conn"))
  {
    DRAY_ERROR("Grid function missing connectivity");
  }

  gf.from_node(n_gf);

  return gf;
}

void validate(const conduit::Node &node, std::vector<std::string> &info)
{

  // info[0] ==  topological dims
  // info[1] == tensor / simplex
  // info[2] == components
  // info[3] == order
  if(!node.has_path("type_name"))
  {
    DRAY_ERROR("Mesh node has no type_name");
  }
  const std::string type_name = node["type_name"].as_string();

  //std::cout<<"Type name "<<type_name<<"\n";
  info = detail::split(type_name, "_");;
  //for(int i = 0; i < info.size(); ++i)
  //{
  //  std::cout<<info[i]<<"\n";
  //}

  if(info[0] != "2D" && info[0] != "3D")
  {
    DRAY_ERROR("Unknown topological dim:'"<<info[0]<<"'");
  }

  if(info[1] != "Simplex" && info[1] != "Tensor")
  {
    DRAY_ERROR("Unknown element type :'"<<info[1]<<"'");
  }

  if(!node.has_path("grid_function"))
  {
    DRAY_ERROR("Mesh missing grid function");
  }

  if(!node.has_path("order"))
  {
    DRAY_ERROR("Missing order");
  }
}

std::shared_ptr<Mesh>
import_mesh(const conduit::Node &n_mesh, std::string mesh_name)
{
  std::shared_ptr<Mesh> res;

  std::vector<std::string> info;
  validate(n_mesh, info);

  int32 order = n_mesh["order"].to_int32();

  const conduit::Node &n_gf = n_mesh["grid_function"];

  if(info[0] == "2D")
  {
    if(info[1] == "Simplex")
    {
      // triangle
    }
    else
    {
      // quad
      //std::cout<<"Quad\n";
      GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);
      using Quad = MeshElem<2u, Tensor, General>;
      using Quad_P1 = MeshElem<2u, Tensor, Linear>;
      using Quad_P2 = MeshElem<2u, Tensor, Quadratic>;

      if(order == 1)
      {
        UnstructuredMesh<Quad_P1> mesh(gf, order);
        res = std::make_shared<QuadMesh_P1>(mesh);
      }
      else if(order == 2)
      {
        UnstructuredMesh<Quad_P2> mesh(gf, order);
        res = std::make_shared<QuadMesh_P2>(mesh);
      }
      else
      {
        UnstructuredMesh<Quad> mesh (gf, order);
        res = std::make_shared<QuadMesh>(mesh);
      }
    }
  }
  else if(info[0] == "3D")
  {
    if(info[1] == "Simplex")
    {
      // tet
    }
    else
    {
      // hex
      //std::cout<<"Hex\n";
      GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);
      using Hex = MeshElem<3u, Tensor, General>;
      using Hex_P1 = MeshElem<3u, Tensor, Linear>;
      using Hex_P2 = MeshElem<3u, Tensor, Quadratic>;

      if(order == 1)
      {
        UnstructuredMesh<Hex_P1> mesh(gf, order);
        res = std::make_shared<HexMesh_P1>(mesh);
      }
      else if(order == 2)
      {
        UnstructuredMesh<Hex_P2> mesh(gf, order);
        res = std::make_shared<HexMesh_P2>(mesh);
      }
      else
      {
        UnstructuredMesh<Hex> mesh (gf, order);
        res = std::make_shared<HexMesh>(mesh);
      }
    }
  }
  res->name(mesh_name);

  return res;
}

void import_field(const conduit::Node &n_field, DataSet &dataset)
{

  const std::string field_name = n_field.name();
  //std::cout<<"Importing field "<<n_field.name()<<"\n";
  std::vector<std::string> info;
  validate(n_field, info);

  int32 order = n_field["order"].to_int32();

  const conduit::Node &n_gf = n_field["grid_function"];
  const int32 phys_dim = n_gf["phys_dim"].to_int32();

  if(info[0] == "2D")
  {
    if(info[1] == "Simplex")
    {
      // triangle
    }
    else
    {
      // quad
      //std::cout<<"Quad\n";
      if(phys_dim == 1)
      {

        GridFunction<1> gf = detail::import_grid_function<1>(n_gf, 1);

        if(order == 1)
        {
          UnstructuredField<QuadScalar_P1> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar_P1>>(field));
        }
        else if(order == 2)
        {
          UnstructuredField<QuadScalar_P2> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar_P2>>(field));
        }
        else
        {
          UnstructuredField<QuadScalar> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadScalar>>(field));
        }
      }
      else if(phys_dim == 3)
      {
        GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);

        if(order == 1)
        {
          UnstructuredField<QuadVector_P1> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector_P1>>(field));
        }
        else if(order == 2)
        {
          UnstructuredField<QuadVector_P2> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector_P2>>(field));
        }
        else
        {
          UnstructuredField<QuadVector> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<QuadVector>>(field));
        }
      }
    }
  }
  else if(info[0] == "3D")
  {
    if(info[1] == "Simplex")
    {
      // tet
    }
    else
    {
      // hex
      //std::cout<<"hex\n";
      if(phys_dim == 1)
      {
        GridFunction<1> gf = detail::import_grid_function<1>(n_gf, 1);

        if(order == 1)
        {
          UnstructuredField<HexScalar_P1> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar_P1>>(field));
        }
        else if(order == 2)
        {
          UnstructuredField<HexScalar_P2> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar_P2>>(field));
        }
        else
        {
          UnstructuredField<HexScalar> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexScalar>>(field));
        }
      }
      else if(phys_dim == 3)
      {
        GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);

        if(order == 1)
        {
          UnstructuredField<HexVector_P1> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector_P1>>(field));
        }
        else if(order == 2)
        {
          UnstructuredField<HexVector_P2> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector_P2>>(field));
        }
        else
        {
          UnstructuredField<HexVector> field (gf, order, field_name);
          dataset.add_field(std::make_shared<UnstructuredField<HexVector>>(field));
        }
      }
    }
  }

}

} // namspace detail

DataSet
to_dataset(const conduit::Node &n_dataset)
{
  if(!n_dataset.has_path("meshes"))
  {
    DRAY_ERROR("Node has no meshes");
  }

  const conduit::Node &n_meshs = n_dataset["meshes"];
  const int32 num_meshes = n_dataset["meshes"].number_of_children();

  DataSet dataset;
  for(int32 i = 0; i < num_meshes; ++i)
  {
    const conduit::Node &n_mesh = n_meshs.child(i);
    dataset.add_mesh(detail::import_mesh(n_mesh, n_mesh.name()));
  }

  if(n_dataset.has_path("fields"))
  {
    const int32 num_fields = n_dataset["fields"].number_of_children();
    for(int32 i = 0; i < num_fields; ++i)
    {
      detail::import_field(n_dataset["fields"].child(i), dataset);
    }
  }

  return dataset;
}

} // namespace dray
