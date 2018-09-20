//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: ascent_catalyst_data_adapter.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_VTK_DATA_ADAPTER_HPP
#define ASCENT_VTK_DATA_ADAPTER_HPP


// forward decs

class vtkDataObject;
class vtkAbstractArray;
class vtkDataArray;
class vtkImageData;
class vtkMultiBlockDataSet;
class vtkRectilinearGrid;
class vtkStructuredGrid;
class vtkUnstructuredGrid;

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Class that Handles Blueprint to VTK Data Transforms
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class VTKDataAdapter
{
public:
    // convert blueprint data to a VTK (which is also a Catalyst) Data Object
    // assumes "n" conforms to the mesh blueprint
    //
    //  conduit::blueprint::mesh::verify(n,info) == true
    //
    // zero copy means attempt to zero copy
    static vtkMultiBlockDataSet* BlueprintToVTKMultiBlock(
      const conduit::Node& n,
      bool zero_copy = false,
      const std::string& topo_name="");

    static vtkDataObject* BlueprintToVTKDataObject(
      const conduit::Node& n,
      bool zero_copy = false,
      const std::string& topo_name="");

private:
    // helpers for specific conversion cases
    static vtkDataObject* UniformBlueprintToVTKDataObject(
      const std::string& coords_name,
      const conduit::Node& n_coords,
      const std::string& topo_name,
      const conduit::Node& n_topo,
      int& neles,
      int& nverts);

    static vtkRectilinearGrid* RectilinearBlueprintToVTKDataObject(
      const std::string& coords_name,
      const conduit::Node& n_coords,
      const std::string& topo_name,
      const conduit::Node& n_topo,
      int& neles,
      int& nverts,
      bool zero_copy);

    static vtkStructuredGrid* StructuredBlueprintToVTKDataObject(
      const std::string& coords_name,
      const conduit::Node& n_coords,
      const std::string& topo_name,
      const conduit::Node& n_topo,
      int& neles,
      int& nverts,
      bool zero_copy);

     static vtkUnstructuredGrid* UnstructuredBlueprintToVTKDataObject(
       const std::string& coords_name,
       const conduit::Node& n_coords,
       const std::string& topo_name,
       const conduit::Node& n_topo,
       int& neles,
       int& nverts,
       bool zero_copy);

    // helper for adding field data
    static void AddField(
      const std::string& field_name,
      const conduit::Node& n_field,
      const std::string& topo_name,
      int neles,
      int nverts,
      vtkDataObject* data,
      bool zero_copy);

    static bool VTKTopologyToBlueprint(
      conduit::Node& output, const vtkDataObject* data_set);

    static void VTKFieldToBlueprint(
      conduit::Node& output, const vtkAbstractArray* field);
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
