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
/// file: ascent_data_adapter.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_VTKH_DATA_ADAPTER_HPP
#define ASCENT_VTKH_DATA_ADAPTER_HPP


// forward decs

// vtkm
namespace vtkm
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
// Class that Handles Blueprint to vtk-h, VTKm Data Transforms
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


    // convert blueprint data to a vtkm Data Set
    // assumes "n" conforms to the mesh blueprint
    //
    //  conduit::blueprint::mesh::verify(n,info) == true
    //
    static vtkm::cont::DataSet  *BlueprintToVTKmDataSet(const conduit::Node &n,
                                                        bool zero_copy,
                                                        const std::string &topo_name);


    // wraps a single VTKm data set into a VTKH dataset
    static vtkh::DataSet    *VTKmDataSetToVTKHDataSet(vtkm::cont::DataSet *dset);

    static void              VTKmToBlueprintDataSet(const vtkm::cont::DataSet *dset,
                                                    conduit::Node &node,
                                                    const std::string topo_name,
                                                    bool zero_copy);

    static void              VTKHToBlueprintDataSet(vtkh::DataSet *dset,
                                                    conduit::Node &node,
                                                    bool zero_copy);

    static void              VTKHCollectionToBlueprintDataSet(VTKHCollection *collection,
                                                              conduit::Node &node,
                                                              bool zero_copy = false);
private:
    // helpers for specific conversion cases
    static vtkm::cont::DataSet  *UniformBlueprintToVTKmDataSet(const std::string &coords_name,
                                                               const conduit::Node &n_coords,
                                                               const std::string &topo_name,
                                                               const conduit::Node &n_topo,
                                                               int &neles,
                                                               int &nverts);


    static vtkm::cont::DataSet  *RectilinearBlueprintToVTKmDataSet(const std::string &coords_name,
                                                                   const conduit::Node &n_coords,
                                                                   const std::string &topo_name,
                                                                   const conduit::Node &n_topo,
                                                                   int &neles,
                                                                   int &nverts,
                                                                   bool zero_copy);

    static vtkm::cont::DataSet  *StructuredBlueprintToVTKmDataSet(const std::string &coords_name,
                                                                  const conduit::Node &n_coords,
                                                                  const std::string &topo_name,
                                                                  const conduit::Node &n_topo,
                                                                  int &neles,
                                                                  int &nverts,
                                                                  bool zero_copy);

     static vtkm::cont::DataSet *UnstructuredBlueprintToVTKmDataSet(const std::string &coords_name,
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
                                          int neles,
                                          int nverts,
                                          vtkm::cont::DataSet *dset,
                                          bool zero_copy);

    static void                  AddVectorField(const std::string &field_name,
                                                const conduit::Node &n_field,
                                                const std::string &topo_name,
                                                int neles,
                                                int nverts,
                                                vtkm::cont::DataSet *dset,
                                                bool zero_copy);

    static bool VTKmTopologyToBlueprint(conduit::Node &output,
                                        const vtkm::cont::DataSet &data_set,
                                        const std::string topo_name,
                                        bool zero_copy);

    static void VTKmFieldToBlueprint(conduit::Node &output,
                                     const vtkm::cont::Field &field,
                                     const std::string topo_name,
                                     bool zero_copy);

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


