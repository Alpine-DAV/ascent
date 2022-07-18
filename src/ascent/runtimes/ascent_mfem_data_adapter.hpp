//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
  int cycle();
  void cycle(int cycle);

  double time();
  void time(double time);
protected:
  FieldMap    m_fields;
  mfem::Mesh *m_mesh;
  int m_cycle;
  double m_time;

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


