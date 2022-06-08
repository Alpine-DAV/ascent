// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/obj_reader.hpp>
#include <dray/rendering/texture2d.hpp>
#include <dray/rendering/material.hpp>
#include <conduit.hpp>
#include <map>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace dray
{


Material import_mat(const tinyobj::material_t &mat,
                    const int32 id,
                    const std::string base_path,
                    std::map<std::string, int32> &texture_ids,
                    std::vector<Texture2d> &textures)
{
  Material dmat;
  dmat.m_id = id;
  dmat.m_ambient[0] = mat.ambient[0];
  dmat.m_ambient[1] = mat.ambient[1];
  dmat.m_ambient[2] = mat.ambient[2];
  dmat.m_diffuse[0] = mat.diffuse[0];
  dmat.m_diffuse[1] = mat.diffuse[1];
  dmat.m_diffuse[2] = mat.diffuse[2];
  dmat.m_specular[0] = mat.specular[0];
  dmat.m_specular[1] = mat.specular[1];
  dmat.m_specular[2] = mat.specular[2];
  // check for the diffuse texture
  const std::string tname = mat.diffuse_texname;
  int tex_id = -1;

  if( tname != "")
  {
    bool loaded = texture_ids.find(tname) != texture_ids.end();
    if(!loaded)
    {
      std::string fpath =
        conduit::utils::join_file_path (base_path, tname);
      tex_id = textures.size();
      Texture2d texture(fpath);
      texture.id(tex_id);
      texture_ids[tname] = tex_id;
      textures.push_back(texture);
    }
    else
    {
      tex_id = texture_ids[tname];
    }
  }

  dmat.m_diff_texture = tex_id;
  return dmat;
}

void print_mat(const tinyobj::material_t &mat)
{
  std::cout<<" material name "<<mat.name<<"\n";  // map_Ka
  std::cout<<" texture amb "<<mat.ambient_texname<<"\n";  // map_Ka
  std::cout<<" texture diff "<<mat.diffuse_texname<<"\n"; // map_Kd
  std::cout<<" texture spec "<<mat.specular_texname<<"\n";    // map_Ks
  std::cout<<" texture high "<<mat.specular_highlight_texname<<"\n";  // map_Ns
  std::cout<<" texture bump "<<mat.bump_texname<<"\n";        // map_bump, map_Bump, bump
  std::cout<<" texture dis "<<mat.displacement_texname<<"\n"; // disp
  std::cout<<" texture alpha "<<mat.alpha_texname<<"\n";      // map_d
  std::cout<<" texture refl "<<mat.reflection_texname<<"\n";        // refl

}

void read_obj (const std::string file_path,
               Array<Vec<float32,3>> &a_verts,
               Array<Vec<int32,3>> &a_indices,
               Array<Vec<float32,2>> &t_coords,
               Array<Vec<int32,3>> &t_indices,
               Array<Material> &a_materials,
               Array<int32> &a_mat_ids,
               std::vector<Texture2d> &textures)
{


  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err,warn;
  std::string file_name, path_b;
  conduit::utils::rsplit_file_path (file_path, file_name, path_b);
  path_b += "/";
  std::string non_const_filename = file_path;
  bool ret = tinyobj::LoadObj (&attrib,
                               &shapes,
                               &materials,
                               &err,
                               &warn,
                               file_path.c_str(),
                               path_b.c_str());

  if (!err.empty ())
  { // `err` may contain warning message.
    std::cerr << err << std::endl;
  }

  if (!ret)
  {
    exit (1);
  }

  for(const auto &mat: materials)
  {
    print_mat(mat);
  }

  const int32 num_verts = attrib.vertices.size () / 3;
  a_verts.resize(num_verts);
  Vec<float32,3> *vert_ptr = a_verts.get_host_ptr();
  for(int i = 0; i < num_verts; ++i)
  {
     const int32 offset = i * 3;
     Vec<float32,3> vert;
     vert[0] = attrib.vertices[offset + 0];
     vert[1] = attrib.vertices[offset + 1];
     vert[2] = attrib.vertices[offset + 2];
     vert_ptr[i] = vert;
  }

  // count the number of triangles
  int tris = 0;
  for (size_t s = 0; s < shapes.size (); s++)
  {
    tris += shapes[s].mesh.num_face_vertices.size ();
  }
  // connectivity
  a_indices.resize (tris);
  Vec<int32,3> *indices = a_indices.get_host_ptr ();

  a_mat_ids.resize(tris);
  int32 *mat_id_ptr = a_mat_ids.get_host_ptr();

  int indices_offset = 0;
  int tri_count = 0;

  bool has_texture = false;
  const int32 t_size = attrib.texcoords.size() / 2;
  if(t_size != 0)
  {
    has_texture = true;
    t_coords.resize(t_size);
    t_indices.resize(tris);
    Vec<float32,2> *tcoords_ptr = t_coords.get_host_ptr();
    for(int32 i = 0; i < t_size; ++i)
    {
      Vec<float32,2> st;
      const int32 offset = i * 2;
      st[0] = attrib.texcoords[offset + 0];
      st[1] = attrib.texcoords[offset + 1];
      tcoords_ptr[i] = st;
    }
  }


  // Loop over shapes
  for (size_t s = 0; s < shapes.size (); s++)
  {
    // Loop over faces(polygon) defults to triangulate
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size (); f++)
    {
      int fv = shapes[s].mesh.num_face_vertices[f];
      // Loop over vertices in the face.
      Vec<int32,3> vindex;
      Vec<int32,3> tindex;
      for (size_t v = 0; v < fv; v++)
      {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        vindex[v] = idx.vertex_index;
        indices_offset++;
        // tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
        // tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
        // tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
         //
         // tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
         // tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
         // tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];

        if(has_texture)
        {
          tindex[v] = idx.texcoord_index;
        }

        //tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
        //tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
        //std::cout<<"texture "<<tx<<" "<<ty<<"\n";
        // Optional: vertex colors
        // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
        // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
        // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
      }
      indices[tri_count] = vindex;
      if(has_texture)
      {
        t_indices.get_host_ptr()[tri_count] = tindex;
      }
      mat_id_ptr[tri_count] = shapes[s].mesh.material_ids[f];
      tri_count++;
      index_offset += fv;

      // per-face material
      // shapes[s].mesh.material_ids[f];
    }
  }

  // set up the materials
  const int32 num_mats = materials.size();
  a_materials.resize(num_mats);
  std::map<std::string, int32> tnames;

  for(int32 i = 0; i < num_mats; i++)
  {
    a_materials.get_host_ptr()[i] = import_mat(materials[i],
                                               i,
                                               path_b,
                                               tnames,
                                               textures);
  }

}

} //namespace dray
