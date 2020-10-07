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
/// file: ascent_mfem_data_adapter.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_mfem_data_adapter.hpp"

#include <ascent_logging.hpp>

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>

// third party includes
#include <conduit_blueprint.hpp>
// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#include <mfem.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

MFEMDataSet::MFEMDataSet()
  : m_cycle(0)
{

}

MFEMDataSet::~MFEMDataSet()
{
  delete m_mesh;

  for(auto it = m_fields.begin(); it != m_fields.end(); ++it)
  {
    delete it->second;
  }
}

MFEMDataSet::MFEMDataSet(mfem::Mesh *mesh)
 : m_mesh(mesh),
   m_cycle(0),
   m_time(0.0)
{

}

int
MFEMDataSet::cycle()
{
  return m_cycle;
}

double
MFEMDataSet::time()
{
  return m_time;
}

void
MFEMDataSet::time(double time)
{
  m_time = time;
}

void
MFEMDataSet::cycle(int cycle)
{
  m_cycle = cycle;
}

void
MFEMDataSet::set_mesh(mfem::Mesh *mesh)
{
  m_mesh = mesh;
}

mfem::Mesh*
MFEMDataSet::get_mesh()
{
  return m_mesh;
}

void
MFEMDataSet::add_field(mfem::GridFunction *field, const std::string &name)
{
  m_fields[name] = field;
}

MFEMDataSet::FieldMap
MFEMDataSet::get_field_map()
{
  return m_fields;
}

bool
MFEMDataSet::has_field(const std::string &field_name)
{
  auto it = m_fields.find(field_name);
  return it != m_fields.end();
}

mfem::GridFunction*
MFEMDataSet::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    std::string msg = "MFEMDataSet: no field named : " + field_name;
    ASCENT_ERROR(msg);
  }

  return m_fields[field_name];
}

int
MFEMDataSet::num_fields()
{
  return m_fields.size();
}

//-----------------------------------------------------------------------------
// MFEMDataAdapter public methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
MFEMDomains*
MFEMDataAdapter::BlueprintToMFEMDataSet(const Node &node,
                                        const std::string &topo_name)
{

    // treat everything as a multi-domain data set

    MFEMDomains *res = new MFEMDomains;


    // get the number of domains and check for id consistency
    const int num_domains = node.number_of_children();


    for(int i = 0; i < num_domains; ++i)
    {
      MFEMDataSet *dset = new MFEMDataSet();
      const conduit::Node &dom = node.child(i);
      // this should exist
      int domain_id = dom["state/domain_id"].to_int();
      // insert domain conversion

      bool zero_copy = true;
      mfem::Mesh *mesh = nullptr;
      mesh = mfem::ConduitDataCollection::BlueprintMeshToMesh(dom, topo_name, zero_copy);
      dset->set_mesh(mesh);

      int cycle = 0;
      if(dom.has_path("state/cycle"))
      {
        cycle = dom["state/cycle"].to_int32();
      }
      dset->cycle(cycle);

      double time = 0;
      if(dom.has_path("state/time"))
      {
        time = dom["state/time"].to_float64();
      }
      dset->time(time);

      std::string t_name = topo_name;
      // no topology name provied, use the first
      if(t_name == "")
      {
        std::vector<std::string> tnames = dom["topologies"].child_names();
        if(tnames.size() > 0)
        {
          t_name = tnames[0];
        }
      }

      if(t_name == "")
      {
        ASCENT_ERROR("Unable to determine topology");
      }

      std::string nodes_gf_name = "";
      const Node &n_topo = dom["topologies/" + t_name];
      if (n_topo.has_child("grid_function"))
      {
        nodes_gf_name = n_topo["grid_function"].as_string();
      }
      if(dom.has_path("fields"))
      {
        const int num_fields = dom["fields"].number_of_children();
        std::vector<std::string> fnames = dom["fields"].child_names();
        for(int f = 0; f < num_fields; ++f)
        {
          const conduit::Node &field = dom["fields"].child(f);

          // skip any field that has a unsupported basis type
          //      (we only supprt H1 (continuos) and L2 (discon)
          bool unsupported = false;
          if(field.has_child("basis"))
          {
            std::string basis = field["basis"].as_string();
            if(basis.find("H1_") == std::string::npos &&
               basis.find("L2_") == std::string::npos)
            {
              unsupported = true;
            }
          }
          // skip mesh nodes gf since they are already processed
          // skip attribute fields, they aren't grid functions
          if ( fnames[f] != nodes_gf_name &&
              fnames[f].find("_attribute") == std::string::npos &&
              !unsupported)
          {
            mfem::GridFunction *gf = mfem::ConduitDataCollection::BlueprintFieldToGridFunction(mesh,
                                                                                               field,
                                                                                               zero_copy);
            dset->add_field(gf, fnames[f]);
          }
        }
      }

      res->m_data_sets.push_back(dset);
      res->m_domain_ids.push_back(domain_id);

    }
    return res;
}

// although is should probably never be the case that on domain
// is high order and the others are not, this method will
// return true if at least one domian is higher order
bool
MFEMDataAdapter::IsHighOrder(const conduit::Node &n)
{

  // treat everything as a multi-domain data set
  const int num_domains = n.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = n.child(i);
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      const int num_fields= fields.number_of_children();
      for(int t = 0; t < num_fields; ++t)
      {
        const conduit::Node &field = fields.child(t);
        if(field.has_path("basis")) return true;
      }

    }
  }

  return false;
}
//                                          VDim
// +------------+--------------------+------------------+
// | Space Type | FE Collection Type | Vector Dimension |
// +------------+--------------------+------------------+
// | SC         | H1_FESpace         | 1                | scalar continous (nodes)
// +------------+--------------------+------------------+
// | SD         | L1                 | 1                | scalar discontinous (zonal sort of)
// +------------+--------------------+------------------+
// | VC         | H1                 | D                | vector continious
// +------------+--------------------+------------------+
// | TC         | SCColl             | (D*(D+1))/2      | tensor continous
// +------------+--------------------+------------------+
// | TD         | SDColl             | (D*(D+1))/2      | tensor discontinous
// +------------+--------------------+------------------+
// | RT         | RTColl             | 1                |
// +------------+--------------------+------------------+
// | ND         | NDColl             | 1                |
// +------------+--------------------+------------------+
void
MFEMDataAdapter::Linearize(MFEMDomains *ho_domains, conduit::Node &output, const int refinement)
{
  const int n_doms = ho_domains->m_data_sets.size();

  output.reset();
  for(int i = 0; i < n_doms; ++i)
  {

    conduit::Node &n_dset = output.append();
    n_dset["state/domain_id"] = int(ho_domains->m_domain_ids[i]);
    n_dset["state/cycle"] = int(ho_domains->m_data_sets[i]->cycle());
    n_dset["state/time"] = double(ho_domains->m_data_sets[i]->time());

    // get the high order data
    mfem::Mesh *ho_mesh = ho_domains->m_data_sets[i]->get_mesh();
    const mfem::FiniteElementSpace *ho_fes_space = ho_mesh->GetNodalFESpace();
    const mfem::FiniteElementCollection *ho_fes_col = ho_fes_space->FEColl();
    // refine the mesh and convert to blueprint
    mfem::Mesh *lo_mesh = new mfem::Mesh(ho_mesh, refinement, mfem::BasisType::GaussLobatto);
    MeshToBlueprintMesh (lo_mesh, n_dset);

    int conn_size = n_dset["topologies/main/elements/connectivity"].dtype().number_of_elements();

    conduit::Node &n_fields = n_dset["fields"];
    auto field_map = ho_domains->m_data_sets[i]->get_field_map();

    for(auto it = field_map.begin(); it != field_map.end(); ++it)
    {

      mfem::GridFunction *ho_gf = it->second;
      std::string basis(ho_gf->FESpace()->FEColl()->Name());
      // we only have L2 or H2 at this point
      bool node_centered = basis.find("H1_") != std::string::npos;

      mfem::FiniteElementSpace *ho_fes = ho_gf->FESpace();
      if(ho_fes == nullptr)
      {
        ASCENT_ERROR("Linearize: high order gf finite element space is null")
      }
      // create the low order grid function
      mfem::FiniteElementCollection *lo_col = nullptr;
      if(node_centered)
      {
        lo_col = new mfem::LinearFECollection;
      }
      else
      {
        int  p = 0; // single scalar
        lo_col = new mfem::L2_FECollection(p, ho_mesh->Dimension(), 1);
      }
      mfem::FiniteElementSpace *lo_fes = new mfem::FiniteElementSpace(lo_mesh, lo_col, ho_fes->GetVDim());
      mfem::GridFunction *lo_gf = new mfem::GridFunction(lo_fes);
      // transform the higher order function to a low order function somehow
      mfem::OperatorHandle hi_to_lo;
      lo_fes->GetTransferOperator(*ho_fes, hi_to_lo);
      hi_to_lo.Ptr()->Mult(*ho_gf, *lo_gf);
      // extract field
      conduit::Node &n_field = n_fields[it->first];
      GridFunctionToBlueprintField(lo_gf, n_field);
      // all supported grid functions coming out of mfem end up being associtated with vertices
      if(node_centered)
      {
        n_field["association"] = "vertex";
      }
      else
      {
        n_field["association"] = "element";
      }

      delete lo_col;
      delete lo_fes;
      delete lo_gf;
    }

    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",n_dset,info);
    if(!success)
    {
      info.print();
      ASCENT_ERROR("Linearize: failed to build a blueprint conforming data set from mfem")
    }
    delete lo_mesh;

  }
  //output.schema().print();
}

void
MFEMDataAdapter::GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                              Node &n_field,
                                              const std::string &main_topology_name)
{
   n_field["basis"] = gf->FESpace()->FEColl()->Name();
   n_field["topology"] = main_topology_name;

   int vdim  = gf->FESpace()->GetVDim();
   int ndofs = gf->FESpace()->GetNDofs();

   const double * values = gf->HostRead();
   if (vdim == 1) // scalar case
   {
      //n_field["values"].set_external(gf->GetData(),
      //                               ndofs);
      n_field["values"].set(values,
                            ndofs);
   }
   else // vector case
   {
      // deal with striding of all components

     mfem::Ordering::Type ordering = gf->FESpace()->GetOrdering();

      int entry_stride = (ordering == mfem::Ordering::byNODES ? 1 : vdim);
      int vdim_stride  = (ordering == mfem::Ordering::byNODES ? ndofs : 1);

      index_t offset = 0;
      index_t stride = sizeof(double) * entry_stride;

      for (int d = 0;  d < vdim; d++)
      {
         std::ostringstream oss;
         oss << "v" << d;
         std::string comp_name = oss.str();
         //n_field["values"][comp_name].set_external(gf->GetData(),
         //                                          ndofs,
         //                                          offset,
         //                                          stride);
         n_field["values"][comp_name].set(values,
                                          ndofs,
                                          offset,
                                          stride);
         offset +=  sizeof(double) * vdim_stride;
      }
   }

}

void
MFEMDataAdapter::MeshToBlueprintMesh(mfem::Mesh *mesh,
                                     Node &n_mesh,
                                     const std::string &coordset_name,
                                     const std::string &main_topology_name,
                                     const std::string &boundary_topology_name)
{
   int dim = mesh->SpaceDimension();

   if(dim < 1 || dim > 3)
   {
     ASCENT_ERROR("invalid mesh dimension "<<dim);;
   }

   ////////////////////////////////////////////
   // Setup main coordset
   ////////////////////////////////////////////

   // Assumes  mfem::Vertex has the layout of a double array.

   // this logic assumes an mfem vertex is always 3 doubles wide
   int stride = sizeof(mfem::Vertex);
   int num_vertices = mesh->GetNV();

   if(stride != 3 * sizeof(double) )
   {
     ASCENT_ERROR("Unexpected stride for mfem vertex");
   }

   Node &n_mesh_coords = n_mesh["coordsets"][coordset_name];
   n_mesh_coords["type"] =  "explicit";


   double *coords_ptr = mesh->GetVertex(0);

   n_mesh_coords["values/x"].set(coords_ptr,
                                 num_vertices,
                                 0,
                                 stride);

   if (dim >= 2)
   {
      n_mesh_coords["values/y"].set(coords_ptr,
                                    num_vertices,
                                    sizeof(double),
                                    stride);
   }
   if (dim >= 3)
   {
      n_mesh_coords["values/z"].set(coords_ptr,
                                    num_vertices,
                                    sizeof(double) * 2,
                                    stride);
   }

   ////////////////////////////////////////////
   // Setup main topo
   ////////////////////////////////////////////

   Node &n_topo = n_mesh["topologies"][main_topology_name];

   n_topo["type"]  = "unstructured";
   n_topo["coordset"] = coordset_name;

   mfem::Element::Type ele_type = static_cast<mfem::Element::Type>(mesh->GetElement(
                                                          0)->GetType());

   std::string ele_shape = ElementTypeToShapeName(ele_type);

   n_topo["elements/shape"] = ele_shape;

   mfem::GridFunction *gf_mesh_nodes = mesh->GetNodes();

   if (gf_mesh_nodes != NULL)
   {
      n_topo["grid_function"] =  "mesh_nodes";
   }

   // connectivity
   // TODO: generic case, i don't think we can zero-copy (mfem allocs
   // an array per element) so we alloc our own contig array and
   // copy out. Some other cases (sidre) may actually have contig
   // allocation but I am  not sure how to detect this case from mfem
   int num_ele = mesh->GetNE();
   int geom = mesh->GetElementBaseGeometry(0);
   int idxs_per_ele = mfem::Geometry::NumVerts[geom];
   int num_conn_idxs =  num_ele * idxs_per_ele;

   n_topo["elements/connectivity"].set(DataType::c_int(num_conn_idxs));

   int *conn_ptr = n_topo["elements/connectivity"].value();

   for (int i=0; i < num_ele; i++)
   {
      const mfem::Element *ele = mesh->GetElement(i);
      const int *ele_verts = ele->GetVertices();

      memcpy(conn_ptr, ele_verts, idxs_per_ele * sizeof(int));

      conn_ptr += idxs_per_ele;
   }

   if (gf_mesh_nodes != NULL)
   {
      GridFunctionToBlueprintField(gf_mesh_nodes,
                                   n_mesh["fields/mesh_nodes"],
                                   main_topology_name);
      n_mesh["fields/mesh_nodes/association"] = "vertex";
   }

   ////////////////////////////////////////////
   // Setup mesh attribute
   ////////////////////////////////////////////

   Node &n_mesh_att = n_mesh["fields/element_attribute"];

   n_mesh_att["association"] = "element";
   n_mesh_att["topology"] = main_topology_name;
   n_mesh_att["values"].set(DataType::c_int(num_ele));

   int_array att_vals = n_mesh_att["values"].value();
   for (int i = 0; i < num_ele; i++)
   {
      att_vals[i] = mesh->GetAttribute(i);
   }

   ////////////////////////////////////////////
   // Setup bndry topo "boundary"
   ////////////////////////////////////////////

   // guard vs if we have boundary elements
   if (mesh->GetNBE() > 0)
   {
      n_topo["boundary_topology"] = boundary_topology_name;

      Node &n_bndry_topo = n_mesh["topologies"][boundary_topology_name];

      n_bndry_topo["type"]     = "unstructured";
      n_bndry_topo["coordset"] = coordset_name;

      mfem::Element::Type bndry_ele_type = static_cast<mfem::Element::Type>(mesh->GetBdrElement(
                                                                   0)->GetType());

      std::string bndry_ele_shape = ElementTypeToShapeName(bndry_ele_type);

      n_bndry_topo["elements/shape"] = bndry_ele_shape;


      int num_bndry_ele = mesh->GetNBE();
      int bndry_geom    = mesh->GetBdrElementBaseGeometry(0);
      int bndry_idxs_per_ele  = mfem::Geometry::NumVerts[bndry_geom];
      int num_bndry_conn_idxs =  num_bndry_ele * bndry_idxs_per_ele;

      n_bndry_topo["elements/connectivity"].set(DataType::c_int(num_bndry_conn_idxs));

      int *bndry_conn_ptr = n_bndry_topo["elements/connectivity"].value();

      for (int i=0; i < num_bndry_ele; i++)
      {
         const mfem::Element *bndry_ele = mesh->GetBdrElement(i);
         const int *bndry_ele_verts = bndry_ele->GetVertices();

         memcpy(bndry_conn_ptr, bndry_ele_verts, bndry_idxs_per_ele  * sizeof(int));

         bndry_conn_ptr += bndry_idxs_per_ele;
      }

      ////////////////////////////////////////////
      // Setup bndry mesh attribute
      ////////////////////////////////////////////

      Node &n_bndry_mesh_att = n_mesh["fields/boundary_attribute"];

      n_bndry_mesh_att["association"] = "element";
      n_bndry_mesh_att["topology"] = boundary_topology_name;
      n_bndry_mesh_att["values"].set(DataType::c_int(num_bndry_ele));

      int_array bndry_att_vals = n_bndry_mesh_att["values"].value();
      for (int i = 0; i < num_bndry_ele; i++)
      {
         bndry_att_vals[i] = mesh->GetBdrAttribute(i);
      }
   }
}
std::string
MFEMDataAdapter::ElementTypeToShapeName(mfem::Element::Type element_type)
{
   // Adapted from SidreDataCollection

   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL,
   //                        TETRAHEDRON, HEXAHEDRON };
   // Note: -- the string names are from conduit's blueprint

   switch (element_type)
   {
     case mfem::Element::POINT:          return "point";
     case mfem::Element::SEGMENT:        return "line";
     case mfem::Element::TRIANGLE:       return "tri";
     case mfem::Element::QUADRILATERAL:  return "quad";
     case mfem::Element::TETRAHEDRON:    return "tet";
     case mfem::Element::HEXAHEDRON:     return "hex";
     case mfem::Element::WEDGE:          return "wedge";
   }

   return "unknown";
}

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
