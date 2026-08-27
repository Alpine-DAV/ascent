//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_data_adapter.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_VTKH_DATA_ADAPTER_HPP
#define ASCENT_VTKH_DATA_ADAPTER_HPP


// forward decs

// viskores
namespace viskores
{
namespace cont
{
class DataSet;
class Field;
};
};

// vtkh
namespace vtkh
{
class DataSet;
};

#include <ascent_exports.h>
#include "ascent_vtkh_collection.hpp"
// conduit includes
#include <conduit.hpp>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Class that Handles Blueprint to vtk-h, Viskores Data Transforms
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class ASCENT_API VTKHDataAdapter
{
public:

    //
    // Convert a multi-domain blueprint data set to a VTKHCollection
    //  assumes: conduit::blueprint::mesh::verify(n,info) == true
    //
    static VTKHCollection* BlueprintToVTKHCollection(const conduit::Node &n,
                                                     bool zero_copy);
    // convert blueprint data to a vtkh Data Set
    // assumes "n" conforms to the mesh blueprint
    //
    //  conduit::blueprint::mesh::verify(n,info) == true
    //
    // zero copy means attempt to zero copy
    static vtkh::DataSet  *BlueprintToVTKHDataSet(const conduit::Node &n,
                                                  const std::string &topo_name,
                                                  bool zero_copy = false);


    // convert blueprint data to a viskores Data Set
    // assumes "n" conforms to the mesh blueprint
    //
    //  conduit::blueprint::mesh::verify(n,info) == true
    //
    static viskores::cont::DataSet  *BlueprintToViskoresDataSet(const conduit::Node &n,
                                                        bool zero_copy,
                                                        const std::string &topo_name);


    // wraps a single Viskores data set into a VTKH dataset
    static vtkh::DataSet    *ViskoresDataSetToVTKHDataSet(viskores::cont::DataSet *dset);

    static void              ViskoresToBlueprintDataSet(const viskores::cont::DataSet *dset,
                                                    conduit::Node &node,
                                                    const std::string &topo_name,
                                                    bool zero_copy);

    static void              VTKHToBlueprintDataSet(vtkh::DataSet *dset,
                                                    conduit::Node &node,
                                                    bool zero_copy);

    static void              VTKHCollectionToBlueprintDataSet(VTKHCollection *collection,
                                                              conduit::Node &node,
                                                              bool zero_copy = false);
private:
    // helpers for specific conversion cases
    static viskores::cont::DataSet  *UniformBlueprintToViskoresDataSet(const std::string &coords_name,
                                                               const conduit::Node &n_coords,
                                                               const std::string &topo_name,
                                                               const conduit::Node &n_topo,
                                                               int &neles,
                                                               int &nverts);


    static viskores::cont::DataSet  *RectilinearBlueprintToViskoresDataSet(const std::string &coords_name,
                                                                   const conduit::Node &n_coords,
                                                                   const std::string &topo_name,
                                                                   const conduit::Node &n_topo,
                                                                   int &neles,
                                                                   int &nverts,
                                                                   bool zero_copy);

    static viskores::cont::DataSet  *StructuredBlueprintToViskoresDataSet(const std::string &coords_name,
                                                                  const conduit::Node &n_coords,
                                                                  const std::string &topo_name,
                                                                  const conduit::Node &n_topo,
                                                                  int &neles,
                                                                  int &nverts,
                                                                  bool zero_copy);

     static viskores::cont::DataSet *PointsImplicitBlueprintToViskoresDataSet(const std::string &coords_name,
                                                                      const conduit::Node &n_coords,
                                                                      const std::string &topo_name,
                                                                      const conduit::Node &n_topo,
                                                                      int &neles,
                                                                      int &nverts,
                                                                      bool zero_copy);

     static viskores::cont::DataSet *UnstructuredBlueprintToViskoresDataSet(const std::string &coords_name,
                                                                    const conduit::Node &n_coords,
                                                                    const std::string &topo_name,
                                                                    const conduit::Node &n_topo,
                                                                    int &neles,
                                                                    int &nverts,
                                                                    bool zero_copy);

    // helper for adding field data
    static void                  AddField(const std::string &field_name,
                                          const conduit::Node &n_field,
                                          const std::string &topo_name,
                                          const conduit::Node &n_topo,
                                          int neles,
                                          int nverts,
                                          viskores::cont::DataSet *dset,
                                          bool zero_copy);

    static void                  AddVectorField(const std::string &field_name,
                                                const conduit::Node &n_field,
                                                const std::string &topo_name,
                                                int neles,
                                                int nverts,
                                                viskores::cont::DataSet *dset,
                                                const int dims,
                                                bool zero_copy);

    static void                  AddMatSets(const std::string &matset_name,
                                            const conduit::Node &n_matset,
                                            const std::string &topo_name,
                                            int neles,
                                            viskores::cont::DataSet *dset,
                                            bool zero_copy);

    static bool ViskoresTopologyToBlueprint(conduit::Node &output,
                                        const viskores::cont::DataSet &data_set,
                                        const std::string &topo_name,
                                        bool zero_copy);

    static void ViskoresFieldToBlueprint(conduit::Node &output,
                                     const viskores::cont::Field &field,
                                     const std::string &topo_name,
                                     bool zero_copy);

    static void ViskoresBlueprintShapeMap(conduit::Node &output);

    static bool CheckShapeMapVsViskoresShapeIds(const conduit::Node &shape_map);


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

