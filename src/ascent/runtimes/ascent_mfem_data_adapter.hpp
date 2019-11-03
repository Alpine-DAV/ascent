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
/// file: ascent_mfem_data_adapter.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_MFEM_DATA_ADAPTER_HPP
#define ASCENT_MFEM_DATA_ADAPTER_HPP


// conduit includes
#include <conduit.hpp>
#include <mfem.hpp>

#include <ascent_exports.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


class ASCENT_API MFEMDataSet
{
public:
  using FieldMap = std::map<std::string, mfem::GridFunction*>;
  MFEMDataSet();
  ~MFEMDataSet();
  MFEMDataSet(mfem::Mesh *mesh);

  void set_mesh(mfem::Mesh *mesh);
  mfem::Mesh* get_mesh();

  void add_field(mfem::GridFunction *field, const std::string &name);
  bool has_field(const std::string &field_name);
  mfem::GridFunction* get_field(const std::string &field_name);
  int num_fields();
  FieldMap get_field_map();
protected:
  FieldMap    m_fields;
  mfem::Mesh *m_mesh;

};

struct ASCENT_API MFEMDomains
{
  std::vector<MFEMDataSet*> m_data_sets;
  std::vector<int> m_domain_ids;
  ~MFEMDomains()
  {
    for(int i = 0; i < m_data_sets.size(); ++i)
    {
      delete m_data_sets[i];
    }
  }
};
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Class that Handles Blueprint to mfem
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class ASCENT_API MFEMDataAdapter
{
public:

    // convert blueprint mfem data to a mfem data set
    // assumes "n" conforms to the mesh blueprint
    //
    //  conduit::blueprint::mesh::verify(n,info) == true
    //
    static MFEMDomains* BlueprintToMFEMDataSet(const conduit::Node &n,
                                               const std::string &topo_name="");

    static bool IsHighOrder(const conduit::Node &n);

    static void Linearize(MFEMDomains *ho_domains, conduit::Node &output, const int refinement);

    static void GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                            conduit::Node &out,
                                            const std::string &main_topology_name = "main");
    static void MeshToBlueprintMesh(mfem::Mesh *m,
                                    conduit::Node &out,
                                    const std::string &coordset_name = "coords",
                                    const std::string &main_topology_name = "main",
                                    const std::string &boundary_topology_name = "boundary");

    static std::string ElementTypeToShapeName(mfem::Element::Type element_type);

};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


