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
/// file: ascent_data_adapter.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_vtkh_data_adapter.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>
#include <type_traits>

// thirdparty includes

// VTKm includes
#define VTKM_USE_DOUBLE_PRECISION
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayHandleExtractComponent.h>
#include <vtkh/DataSet.hpp>
// other ascent includes
#include <ascent_logging.hpp>
#include <ascent_block_timer.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>

#include <conduit_blueprint.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// VTKHDataAdapter public methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
vtkh::DataSet *
VTKHDataAdapter::BlueprintToVTKHDataSet(const Node &node,
                                        const std::string &topo_name)
{       
 
    // treat everything as a multi-domain data set 
    conduit::Node multi_dom; 
    blueprint::mesh::to_multi_domain(node, multi_dom);

    vtkh::DataSet *res = new vtkh::DataSet;


    int num_domains = 0;
    bool has_ids = true;
    bool no_ids = true;
  
    // get the number of domains and check for id consistency
    num_domains = multi_dom.number_of_children();

    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = multi_dom.child(i);
      if(dom.has_path("state/domain_id"))
      {
        no_ids = false; 
      }
      else
      {
        has_ids = false;
      }
    }
#ifdef ASCENT_MPI_ENABLED
    int comm_size = vtkh::GetMPISize();
    int *has_ids_array = new int[comm_size];
    int *no_ids_array = new int[comm_size];
    int boolean = has_ids ? 1 : 0; 
    MPI_Allgather(&boolean, 1, MPI_INT, has_ids_array, 1, MPI_INT, vtkh::GetMPIComm());
    boolean = no_ids ? 1 : 0; 
    MPI_Allgather(&boolean, 1, MPI_INT, no_ids_array, 1, MPI_INT, vtkh::GetMPIComm());

    bool global_has_ids = true;
    bool global_no_ids = false;
    for(int i = 0; i < comm_size; ++i)
    {
      if(has_ids_array[i] == 0)
      {
        global_has_ids = false;
      }
      if(no_ids_array[i] == 1)
      {
        global_no_ids = true;
      }
    }
    has_ids = global_has_ids;
    no_ids = global_no_ids;
    delete[] has_ids_array;
    delete[] no_ids_array;
#endif
      
    bool consistent_ids = (has_ids || no_ids);
     
    if(!consistent_ids)
    {
      ASCENT_ERROR("Inconsistent domain ids: all domains must either have an id "
                  <<"or all domains do not have an id");
    }

    int domain_offset = 0;
#ifdef ASCENT_MPI_ENABLED
    int *domains_per_rank = new int[comm_size];
    int rank = vtkh::GetMPIRank();
    MPI_Allgather(&num_domains, 1, MPI_INT, domains_per_rank, 1, MPI_INT, vtkh::GetMPIComm());
    for(int i = 0; i < rank; ++i)
    {
      domain_offset += domains_per_rank[i];
    }
    delete[] domains_per_rank;  
#endif
    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = multi_dom.child(i);      
      vtkm::cont::DataSet *dset = VTKHDataAdapter::BlueprintToVTKmDataSet(dom,
                                                                          topo_name);
      int domain_id = domain_offset;
      if(node.has_path("state/domain_id"))
      {
          domain_id = node["state/domain_id"].to_int();
      }
#ifdef ASCENT_MPI_ENABLED
      else
      {
         domain_id = domain_offset + i;
      }
#endif
      if(node.has_path("state/cycle"))
      {
        vtkm::UInt64 cycle = node["state/cycle"].to_uint64();
        res->SetCycle(cycle);
      }

      res->AddDomain(*dset,domain_id);
      // vtk-m will shallow copy the data assoced with dset
      // clean up our copy
      delete dset;
    
    }    
    return res;
}

//-----------------------------------------------------------------------------
vtkh::DataSet *
VTKHDataAdapter::VTKmDataSetToVTKHDataSet(vtkm::cont::DataSet *dset)
{
    // wrap a single VTKm data set into a VTKH dataset
    vtkh::DataSet   *res = new  vtkh::DataSet;
    int domain_id = 0; // TODO, MPI_TASK_ID ?
    res->AddDomain(*dset,domain_id);
    return res;
}

//-----------------------------------------------------------------------------
vtkm::cont::DataSet *
VTKHDataAdapter::BlueprintToVTKmDataSet(const Node &node,
                                        const std::string &topo_name_str)
{   
    vtkm::cont::DataSet * result = NULL;

    std::string topo_name = topo_name_str;
    // if we don't specify a topology, find the first topology ...
    if(topo_name == "")
    {
        NodeConstIterator itr = node["topologies"].children();
        itr.next();
        topo_name = itr.name();
    }
    else
    {
        if(!node["topologies"].has_child(topo_name))
        {
            ASCENT_ERROR("Invalid topology name: " << topo_name);
        }
    }

    // as long as mesh blueprint verify true, we access data without fear.
    
    const Node &n_topo   = node["topologies"][topo_name];
    string mesh_type     = n_topo["type"].as_string();
    
    string coords_name   = n_topo["coordset"].as_string();
    const Node &n_coords = node["coordsets"][coords_name];

    int neles  = 0;
    int nverts = 0;

    if( mesh_type ==  "uniform")
    {
        result = UniformBlueprintToVTKmDataSet(coords_name,
                                               n_coords,
                                               topo_name,
                                               n_topo,
                                               neles,
                                               nverts);
    }
    else if(mesh_type == "rectilinear")
    {
        result = RectilinearBlueprintToVTKmDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts);
        
    }
    else if(mesh_type == "structured")
    {
        result =  StructuredBlueprintToVTKmDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts);
    }
    else if( mesh_type ==  "unstructured")
    {
        result =  UnstructuredBlueprintToVTKmDataSet(coords_name,
                                                     n_coords,
                                                     topo_name,
                                                     n_topo,
                                                     neles,
                                                     nverts);
    }
    else
    {
        ASCENT_ERROR("Unsupported topology/type:" << mesh_type);
    }
    
    
    if(node.has_child("fields"))
    {
        // add all of the fields:
        NodeConstIterator itr = node["fields"].children();
        while(itr.has_next())
        {
        
            const Node &n_field = itr.next();
            std::string field_name = itr.name();

            // skip vector fields for now, we need to add
            // more logic to AddField
            if(n_field["values"].number_of_children() == 0 )
            {
            
                AddField(field_name,
                         n_field,
                         topo_name,
                         neles,
                         nverts,
                         result);
            }
        }
    }
    return result;
}


//-----------------------------------------------------------------------------
class ExplicitArrayHelper
{
public:
// Helper function to create explicit coordinate arrays for vtkm data sets
void CreateExplicitArrays(vtkm::cont::ArrayHandle<vtkm::UInt8> &shapes,
                          vtkm::cont::ArrayHandle<vtkm::IdComponent> &num_indices,
                          const std::string &shape_type,
                          const vtkm::Id &conn_size,
                          vtkm::IdComponent &dimensionality,
                          int &neles)
{
    vtkm::UInt8 shape_id = 0;
    vtkm::IdComponent indices= 0;
    if(shape_type == "tri")
    {
        shape_id = 3;
        indices = 3; 
        // note: vtkm cell dimensions are topological
        dimensionality = 2; 
    }
    else if(shape_type == "quad")
    {
        shape_id = 9;
        indices = 4; 
        // note: vtkm cell dimensions are topological
        dimensionality = 2; 
    }
    else if(shape_type == "tet")
    {
        shape_id = 10;
        indices = 4; 
        dimensionality = 3; 
    }
    else if(shape_type == "hex")
    {
        shape_id = 12;
        indices = 8;
        dimensionality = 3; 
    }
    // TODO: Not supported in blueprint yet ... 
    // else if(shape_type == "wedge")
    // {
    //     shape_id = 13;
    //     indices = 6;
    //     dimensionality = 3;
    // }
    // else if(shape_type == "pyramid")
    // {
    //     shape_id = 14;
    //     indices = 5;
    //     dimensionality = 3;
    // }
    else
    {
        ASCENT_ERROR("Unsupported element shape " << shape_type);
    }

    if(conn_size < indices) 
        ASCENT_ERROR("Connectivity array size " <<conn_size << " must be at least size " << indices);
    if(conn_size % indices != 0) 
        ASCENT_ERROR("Connectivity array size " <<conn_size << " be evenly divided by indices size" << indices);

    const vtkm::Id num_shapes = conn_size / indices;

    neles = num_shapes;

    shapes.Allocate(num_shapes);
    num_indices.Allocate(num_shapes);

    // We could memset these and then zero copy them but that 
    // would make us responsible for the data. If we just create
    // them, smart pointers will automatically delete them.
    // Hopefull the compiler turns this into a memset.

    const vtkm::UInt8 shape_value = shape_id;
    const vtkm::IdComponent indices_value = indices;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parrallel for
#endif
    for (int i = 0; i < num_shapes; ++i)
    {
        shapes.GetPortalControl().Set(i, shape_value);
        num_indices.GetPortalControl().Set(i, indices_value);
    }
}
};
//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::UniformBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name 
     const Node &n_coords,           // input mesh bp coordset (assumed uniform)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    //
    // blueprint uniform coord set provides:
    //
    //  dims/{i,j,k}
    //  origin/{x,y,z} (optional)
    //  spacing/{dx,dy,dz} (optional)

    //Create implicit vtkm coordinate system
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();
    
    const Node &n_dims = n_coords["dims"];
    
    int dims_i = n_dims["i"].to_int();
    int dims_j = n_dims["j"].to_int();
    int dims_k = 1;

    bool is_2d = true;

    // check for 3d
    if(n_dims.has_path("k"))
    {
        dims_k = n_dims["k"].to_int();
        is_2d = false;
    }
    
    

    float64 origin_x = 0.0;
    float64 origin_y = 0.0;
    float64 origin_z = 0.0;


    float64 spacing_x = 1.0;
    float64 spacing_y = 1.0;
    float64 spacing_z = 1.0;


    if(n_coords.has_child("origin"))
    {
        const Node &n_origin = n_coords["origin"];
        
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
        const Node &n_spacing = n_coords["spacing"];
        
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
    
    // todo, should this be float64 -- or should we read float32 above?
    
    vtkm::Vec<vtkm::Float32,3> origin(origin_x,
                                      origin_y,
                                      origin_z);
    
    vtkm::Vec<vtkm::Float32,3> spacing(spacing_x,
                                       spacing_y,
                                       spacing_z);
    
    vtkm::Id3 dims(dims_i,
                   dims_j,
                   dims_k);
    
    // todo, use actually coordset and topo names?
    result->AddCoordinateSystem( vtkm::cont::CoordinateSystem(coords_name.c_str(),
                                                              dims,
                                                              origin,
                                                              spacing));
    if(is_2d)
    {
      vtkm::Id2 dims2(dims[0], dims[1]); 
      vtkm::cont::CellSetStructured<2> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(dims2);
      result->AddCellSet(cell_set);
    }
    else
    {
      vtkm::cont::CellSetStructured<3> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(dims);
      result->AddCellSet(cell_set);
    }

    neles =  (dims_i - 1) * (dims_j - 1);
    if(dims_k > 1)
    {
        neles *= (dims_k - 1);
    }
    
    nverts =  dims_i * dims_j;
    if(dims_k > 1)
    {
        nverts *= dims_k;
    }

    return result;
}


//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::RectilinearBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name 
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    int x_npts = n_coords["values/x"].dtype().number_of_elements();
    int y_npts = n_coords["values/y"].dtype().number_of_elements();
    int z_npts = 0;

    int32 ndims = 2;
    
    // todo assumes float64
    const float64 *x_coords_ptr = n_coords["values/x"].as_float64_ptr();
    const float64 *y_coords_ptr = n_coords["values/y"].as_float64_ptr();
    const float64 *z_coords_ptr = NULL;
    
    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        z_npts = n_coords["values/z"].dtype().number_of_elements();
        z_coords_ptr = n_coords["values/z"].as_float64_ptr();
    }

    vtkm::cont::ArrayHandle<vtkm::Float64> x_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> y_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> z_coords_handle;
    
    x_coords_handle = vtkm::cont::make_ArrayHandle(x_coords_ptr, x_npts);
    y_coords_handle = vtkm::cont::make_ArrayHandle(y_coords_ptr, y_npts);

    if(ndims == 3)
    {
        z_coords_handle = vtkm::cont::make_ArrayHandle(z_coords_ptr, z_npts);
    }
    else
    {
        z_coords_handle.Allocate(1);
        z_coords_handle.GetPortalControl().Set(0, 0.0);
    }

    static_assert(std::is_same<vtkm::FloatDefault, double>::value,
                  "VTK-m needs to be configured with 'VTKm_USE_DOUBLE_PRECISION=ON'");
    vtkm::cont::ArrayHandleCartesianProduct<
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault> > coords;

    coords = vtkm::cont::make_ArrayHandleCartesianProduct(x_coords_handle,
                                                          y_coords_handle,
                                                          z_coords_handle);

    vtkm::cont::CoordinateSystem coordinate_system(coords_name.c_str(),
                                                  coords);
    result->AddCoordinateSystem(coordinate_system);

    if (ndims == 2)
    {
      vtkm::cont::CellSetStructured<2> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(vtkm::make_Vec(x_npts,
                                                 y_npts));
      result->AddCellSet(cell_set);
    }
    else
    {
      vtkm::cont::CellSetStructured<3> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(vtkm::make_Vec(x_npts,
                                                 y_npts,
                                                 z_npts));
      result->AddCellSet(cell_set);
    }

    nverts = x_npts * y_npts;
    neles = (x_npts - 1) * (y_npts - 1);
    if(ndims > 2)
    {
        nverts *= z_npts;
        neles *= (z_npts - 1);   
    }

    return result;
}

//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::StructuredBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name 
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();
    

    int32 ndims = 2;
    
    const float64 *x_coords_ptr = n_coords["values/x"].as_float64_ptr();
    const float64 *y_coords_ptr = n_coords["values/y"].as_float64_ptr();
    const float64 *z_coords_ptr = NULL;
    
    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        z_coords_ptr = n_coords["values/z"].as_float64_ptr();
    }

    vtkm::cont::ArrayHandle<vtkm::Float64> x_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> y_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> z_coords_handle;
    
    x_coords_handle = vtkm::cont::make_ArrayHandle(x_coords_ptr, nverts);
    y_coords_handle = vtkm::cont::make_ArrayHandle(y_coords_ptr, nverts);

    if(ndims == 3)
    {
        z_coords_handle = vtkm::cont::make_ArrayHandle(z_coords_ptr, nverts);
    }
    else 
    {
        z_coords_handle.Allocate(nverts); 
        // This does not get initialized to zero
        vtkm::Float64 *z = vtkh::GetVTKMPointer(z_coords_handle);
        memset(z, 0.0, nverts * sizeof(vtkm::Float64));
    }
    result->AddCoordinateSystem(
      vtkm::cont::CoordinateSystem(coords_name.c_str(),
        make_ArrayHandleCompositeVector(x_coords_handle,
                                        0,
                                        y_coords_handle,
                                        0,
                                        z_coords_handle,
                                        0)));

    int32 x_elems = n_topo["elements/dims/i"].as_int32(); 
    int32 y_elems = n_topo["elements/dims/j"].as_int32(); 
    if (ndims == 2)
    {
      vtkm::cont::CellSetStructured<2> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(vtkm::make_Vec(x_elems+1,
                                                 y_elems+1));
      result->AddCellSet(cell_set);
      neles = x_elems * y_elems;
    }
    else
    {
      int32 z_elems = n_topo["elements/dims/k"].as_int32(); 
      vtkm::cont::CellSetStructured<3> cell_set(topo_name.c_str());
      cell_set.SetPointDimensions(vtkm::make_Vec(x_elems+1,
                                                 y_elems+1,
                                                 z_elems+1));
      result->AddCellSet(cell_set);
      neles = x_elems * y_elems * z_elems;

    }
    return result;
}



//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::UnstructuredBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name 
     const Node &n_coords,           // input mesh bp coordset (assumed unstructured)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();
    

    int32 ndims = 2;
    
    const float64 *x_coords_ptr = n_coords["values/x"].as_float64_ptr();
    const float64 *y_coords_ptr = n_coords["values/y"].as_float64_ptr();
    const float64 *z_coords_ptr = NULL;
    
    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        z_coords_ptr = n_coords["values/z"].as_float64_ptr();
    }

    vtkm::cont::ArrayHandle<vtkm::Float64> x_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> y_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> z_coords_handle;
    
    x_coords_handle = vtkm::cont::make_ArrayHandle(x_coords_ptr, nverts);
    y_coords_handle = vtkm::cont::make_ArrayHandle(y_coords_ptr, nverts);

    if(ndims == 3)
    {
        z_coords_handle = vtkm::cont::make_ArrayHandle(z_coords_ptr, nverts);
    }
    else 
    {
        z_coords_handle.Allocate(nverts); 
        // This does not get initialized to zero
        for(int i = 0; i < nverts; ++i)
            z_coords_handle.GetPortalControl().Set(i,0.0);
    }
    result->AddCoordinateSystem(
      vtkm::cont::CoordinateSystem(coords_name.c_str(),
        make_ArrayHandleCompositeVector(x_coords_handle,
                                        0,
                                        y_coords_handle,
                                        0,
                                        z_coords_handle,
                                        0)));


    // shapes, number of indices, and connectivity.
    // Will have to do something different if this is a "zoo"

    // TODO: there is a special data set type for single cell types

    const Node &n_topo_eles = n_topo["elements"];
    std::string ele_shape = n_topo_eles["shape"].as_string();

    // TODO: assumes int32, and contiguous
    const int32 *ele_idx_ptr = n_topo_eles["connectivity"].value();
    int32 conn_size = n_topo_eles["connectivity"].dtype().number_of_elements();
    static_assert(std::is_same<vtkm::Id, int>::value,
                  "VTK-m needs to be configured with 'VTKm_USE_64_BIT_IDS=OFF'");
    vtkm::cont::ArrayHandle<vtkm::Id> connectivity = vtkm::cont::make_ArrayHandle(ele_idx_ptr,
                                                                                  conn_size);
    
    vtkm::cont::ArrayHandle<vtkm::UInt8> shapes;
    vtkm::cont::ArrayHandle<vtkm::IdComponent> num_indices;
    vtkm::IdComponent topo_dimensionality;
    ExplicitArrayHelper array_creator;
    array_creator.CreateExplicitArrays(shapes,
                                       num_indices,
                                       ele_shape,
                                       conn_size,
                                       topo_dimensionality,
                                       neles);
    
    vtkm::cont::CellSetExplicit<> cell_set(topo_name.c_str());

    cell_set.Fill(nverts, shapes, num_indices, connectivity);
    
    result->AddCellSet(cell_set);
    
    ASCENT_INFO("neles "  << neles);
    
    return result;
}

//-----------------------------------------------------------------------------

void
VTKHDataAdapter::AddField(const std::string &field_name,
                          const Node &n_field,
                          const std::string &topo_name,
                          int neles,
                          int nverts,
                          vtkm::cont::DataSet *dset)
{
    ASCENT_INFO("nverts "  << nverts);
    ASCENT_INFO("neles "  << neles);
    
    
    // TODO: how do we deal with vector valued fields?, these will be mcarrays
    
    const float64 *values_ptr = n_field["values"].as_float64_ptr();
    string assoc              = n_field["association"].as_string();

    try
    {
        if(assoc == "vertex")
        {
            //This is the method for zero copy
            vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr = vtkm::cont::make_ArrayHandle(values_ptr, nverts);
            dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                             vtkm::cont::Field::ASSOC_POINTS,
                                             vtkm_arr));
        }
        else if( assoc == "element")
        {
            // double minv = 1e24;
            // double maxv = -minv;
            // for(int i = 0; i < neles; ++i)
            // {
            //     double v = values_ptr[i];
            //     minv = std::min(minv,v);
            //     maxv= std::max(maxv,v);
            // }
            //
            // std::cout<<"Min "<<minv<<" max "<<maxv<<"\n";
            
            //This is the method for zero copy
            vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr = vtkm::cont::make_ArrayHandle(values_ptr, neles);
            dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                             vtkm::cont::Field::ASSOC_CELL_SET,
                                             topo_name.c_str(),
                                             vtkm_arr));
        }
    }
    catch (vtkm::cont::Error error)
    {
        ASCENT_ERROR("VTKm exception:" << error.GetMessage());
    }

}

std::string 
GetBlueprintCellName(vtkm::UInt8 shape_id)
{
  std::string name;
  if(shape_id == vtkm::CELL_SHAPE_TRIANGLE)
  {
    name = "tri";  
  }
  else if(shape_id == vtkm::CELL_SHAPE_VERTEX)
  {
    name = "point";
  }
  else if(shape_id == vtkm::CELL_SHAPE_LINE)
  {
    name = "line";
  }
  else if(shape_id == vtkm::CELL_SHAPE_POLYGON)
  {
    ASCENT_ERROR("Polygon is not supported in blueprint");
  }
  else if(shape_id == vtkm::CELL_SHAPE_QUAD)
  {
    name = "quad";
  }
  else if(shape_id == vtkm::CELL_SHAPE_TETRA)
  {
    name = "tet";
  }
  else if(shape_id == vtkm::CELL_SHAPE_HEXAHEDRON)
  {
    name = "hex";
  }
  else if(shape_id == vtkm::CELL_SHAPE_WEDGE)
  {
    ASCENT_ERROR("Wedge is not supported in blueprint");
  }
  else if(shape_id == vtkm::CELL_SHAPE_PYRAMID)
  {
    ASCENT_ERROR("Pyramid is not supported in blueprint");
  }
  return name;
}

void 
VTKHDataAdapter::VTKmTopologyToBlueprint(conduit::Node &output,
                                         const vtkm::cont::DataSet &data_set)
{
  
  const int default_cell_set = 0; 
  int topo_dims;
  bool is_structured = vtkh::VTKMDataSetInfo::IsStructured(data_set, topo_dims, default_cell_set);
  bool is_uniform = vtkh::VTKMDataSetInfo::IsUniform(data_set);
  bool is_rectilinear = vtkh::VTKMDataSetInfo::IsRectilinear(data_set); 

  vtkm::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  if(is_uniform)
  {
    auto points = coords.GetData().Cast<vtkm::cont::ArrayHandleUniformPointCoordinates>();
    auto portal = points.GetPortalConstControl();

    auto origin = portal.GetOrigin();
    auto spacing = portal.GetSpacing();
    auto dims = portal.GetDimensions();
    output["topologies/topo/coordset"] = "coords";
    output["topologies/topo/type"] = "uniform";

    output["coordsets/coords/type"] = "uniform";  
    output["coordsets/coords/dims/i"] = (int) dims[0];  
    output["coordsets/coords/dims/j"] = (int) dims[1];  
    output["coordsets/coords/dims/k"] = (int) dims[2];  
    output["coordsets/coords/origin/x"] = (int) origin[0];  
    output["coordsets/coords/origin/y"] = (int) origin[1];  
    output["coordsets/coords/origin/z"] = (int) origin[2];  
    output["coordsets/coords/spacing/x"] = (int) spacing[0];  
    output["coordsets/coords/spacing/y"] = (int) spacing[1];  
    output["coordsets/coords/spacing/z"] = (int) spacing[2];  
  }
  else if(is_rectilinear)
  {
    typedef vtkm::cont::ArrayHandleCartesianProduct<vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                    vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                    vtkm::cont::ArrayHandle<vtkm::FloatDefault>> Cartesian;
                                              
    typedef vtkm::cont::ArrayHandle<vtkm::FloatDefault> HandleType;
    typedef typename HandleType::template ExecutionTypes<vtkm::cont::DeviceAdapterTagSerial>::PortalConst PortalType;
    typedef typename vtkm::cont::ArrayPortalToIterators<PortalType>::IteratorType IteratorType;

    const auto points = coords.GetData().Cast<Cartesian>();
    auto portal = points.GetPortalConstControl();
    auto x_portal = portal.GetFirstPortal(); 
    auto y_portal = portal.GetSecondPortal(); 
    auto z_portal = portal.GetThirdPortal();

    IteratorType x_iter = vtkm::cont::ArrayPortalToIterators<PortalType>(x_portal).GetBegin();
    IteratorType y_iter = vtkm::cont::ArrayPortalToIterators<PortalType>(y_portal).GetBegin();
    IteratorType z_iter = vtkm::cont::ArrayPortalToIterators<PortalType>(z_portal).GetBegin();
    // work around for conduit not accepting const pointers
    vtkm::FloatDefault *x_ptr = const_cast<vtkm::FloatDefault*>(&(*x_iter));
    vtkm::FloatDefault *y_ptr = const_cast<vtkm::FloatDefault*>(&(*y_iter));
    vtkm::FloatDefault *z_ptr = const_cast<vtkm::FloatDefault*>(&(*z_iter));

    output["topologies/topo/coordset"] = "coords";
    output["topologies/topo/type"] = "rectilinear";

    output["coordsets/coords/type"] = "rectilinear";  
    output["coordsets/coords/values/x"].set(x_ptr, x_portal.GetNumberOfValues()); 
    output["coordsets/coords/values/y"].set(y_ptr, y_portal.GetNumberOfValues());  
    output["coordsets/coords/values/z"].set(z_ptr, z_portal.GetNumberOfValues());  
  }
  else
  {
    int point_dims[3];
    //
    // This still could be structured, but this will always 
    // have an explicit coordinate system
    output["coordsets/coords/type"] = "explicit";  
    using Coords32 = vtkm::cont::ArrayHandleCompositeVectorType<vtkm::cont::ArrayHandle<vtkm::Float32>,
                                                                vtkm::cont::ArrayHandle<vtkm::Float32>,
                                                                vtkm::cont::ArrayHandle<vtkm::Float32>>::type;

    using Coords64 = vtkm::cont::ArrayHandleCompositeVectorType<vtkm::cont::ArrayHandle<vtkm::Float64>,
                                                                vtkm::cont::ArrayHandle<vtkm::Float64>,
                                                                vtkm::cont::ArrayHandle<vtkm::Float64>>::type;
    
    using CoordsVec32 = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>;
    using CoordsVec64 = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>;

    bool coords_32 = coords.GetData().IsSameType(Coords32()); 
                                            

    if(coords.GetData().IsSameType(Coords32()))
    {
      Coords32 points = coords.GetData().Cast<Coords32>();

      auto x_handle = points.GetStorage().GetArrays().GetParameter<1>();
      auto y_handle = points.GetStorage().GetArrays().GetParameter<2>();
      auto z_handle = points.GetStorage().GetArrays().GetParameter<3>();

      point_dims[0] = x_handle.GetNumberOfValues();
      point_dims[1] = y_handle.GetNumberOfValues();
      point_dims[2] = z_handle.GetNumberOfValues();
      output["coordsets/coords/values/x"].set(vtkh::GetVTKMPointer(x_handle), point_dims[0]); 
      output["coordsets/coords/values/y"].set(vtkh::GetVTKMPointer(y_handle), point_dims[1]);  
      output["coordsets/coords/values/z"].set(vtkh::GetVTKMPointer(z_handle), point_dims[2]);  

    }
    else if(coords.GetData().IsSameType(CoordsVec32()))
    {
      CoordsVec32 points = coords.GetData().Cast<CoordsVec32>();

      const int num_vals = points.GetNumberOfValues();
      vtkm::Float32 *points_ptr = (vtkm::Float32*)vtkh::GetVTKMPointer(points);
      const int byte_size = sizeof(vtkm::Float32);

      output["coordsets/coords/values/x"].set(points_ptr, 
                                              num_vals,
                                              byte_size*0,  // byte offset
                                              byte_size*3); // stride 
      output["coordsets/coords/values/y"].set(points_ptr, 
                                              num_vals,
                                              byte_size*1,  // byte offset
                                              sizeof(vtkm::Float32)*3); // stride 
      output["coordsets/coords/values/z"].set(points_ptr, 
                                              num_vals,
                                              byte_size*2,  // byte offset
                                              byte_size*3); // stride 

    }
    else if(coords.GetData().IsSameType(Coords64()))
    {
      Coords64 points = coords.GetData().Cast<Coords64>();

      auto x_handle = points.GetStorage().GetArrays().GetParameter<1>();
      auto y_handle = points.GetStorage().GetArrays().GetParameter<2>();
      auto z_handle = points.GetStorage().GetArrays().GetParameter<3>();

      point_dims[0] = x_handle.GetNumberOfValues();
      point_dims[1] = y_handle.GetNumberOfValues();
      point_dims[2] = z_handle.GetNumberOfValues();
      output["coordsets/coords/values/x"].set(vtkh::GetVTKMPointer(x_handle), point_dims[0]); 
      output["coordsets/coords/values/y"].set(vtkh::GetVTKMPointer(y_handle), point_dims[1]);  
      output["coordsets/coords/values/z"].set(vtkh::GetVTKMPointer(z_handle), point_dims[2]);  

    }
    else if(coords.GetData().IsSameType(CoordsVec64()))
    {
      CoordsVec64 points = coords.GetData().Cast<CoordsVec64>();

      const int num_vals = points.GetNumberOfValues();
      vtkm::Float64 *points_ptr = (vtkm::Float64*)vtkh::GetVTKMPointer(points);
      const int byte_size = sizeof(vtkm::Float64);

      output["coordsets/coords/values/x"].set(points_ptr, 
                                              num_vals,
                                              byte_size*0,  // byte offset
                                              byte_size*3); // stride 
      output["coordsets/coords/values/y"].set(points_ptr, 
                                              num_vals,
                                              byte_size*1,  // byte offset
                                              byte_size*3); // stride 
      output["coordsets/coords/values/z"].set(points_ptr, 
                                              num_vals,
                                              byte_size*2,  // byte offset
                                              byte_size*3); // stride 

    }
    else
    {
      coords.PrintSummary(std::cerr);
      ASCENT_ERROR("Unknown coords type");
    }

    if(is_structured)
    {
      output["topologies/topo/coordset"] = "coords";
      output["topologies/topo/type"] = "structured";
      output["topologies/topo/elements/dims/i"] = (int) point_dims[0];
      output["topologies/topo/elements/dims/j"] = (int) point_dims[1];
      output["topologies/topo/elements/dims/k"] = (int) point_dims[2];
    }
    else
    {
      output["topologies/topo/coordset"] = "coords";
      output["topologies/topo/type"] = "unstructured";
      vtkm::cont::DynamicCellSet dyn_cells = data_set.GetCellSet();

      using SingleType = vtkm::cont::CellSetSingleType<>;
      using MixedType = vtkm::cont::CellSetExplicit<>;

      if(dyn_cells.IsSameType(SingleType()))
      {
        SingleType cells = dyn_cells.Cast<SingleType>(); 
        vtkm::UInt8 shape_id = cells.GetCellShape(0);
        std::string conduit_name = GetBlueprintCellName(shape_id); 
        output["topologies/topo/elements/shape"] = conduit_name;

        static_assert(sizeof(vtkm::Id) == sizeof(int), "blueprint expects connectivity to be ints"); 
        auto conn = cells.GetConnectivityArray(vtkm::TopologyElementTagPoint(), 
                                               vtkm::TopologyElementTagCell());

        output["topologies/topo/elements/connectivity"].set(vtkh::GetVTKMPointer(conn), 
                                                             conn.GetNumberOfValues());
      }
      else
      {
        ASCENT_ERROR("Mixed explicit types not implemented");
        MixedType cells = dyn_cells.Cast<MixedType>(); 
      }
          
    }
  }
}

template<typename T, int N>
void ConvertVecToNode(conduit::Node &output, 
                      std::string path, 
                      vtkm::cont::ArrayHandle<vtkm::Vec<T,N>> &handle)
{
  static_assert(N > 1 && N < 4, "Vecs must be size 2 or 3");
  output[path + "/type"] = "vector"; 
  output[path + "/values/u"].set((T*) vtkh::GetVTKMPointer(handle), 
                                 handle.GetNumberOfValues(),
                                 sizeof(T)*0,   // starting offset in bytes
                                 sizeof(T)*N);  // stride in bytes 
  output[path + "/values/v"].set((T*) vtkh::GetVTKMPointer(handle), 
                                 handle.GetNumberOfValues(),
                                 sizeof(T)*1,   // starting offset in bytes
                                 sizeof(T)*N);  // stride in bytes 
  if(N == 3)
  {
    
    output[path + "/values/w"].set((T*) vtkh::GetVTKMPointer(handle), 
                                   handle.GetNumberOfValues(),
                                   sizeof(T)*2,   // starting offset in bytes
                                   sizeof(T)*N);  // stride in bytes 
  }
}

void 
VTKHDataAdapter::VTKmFieldToBlueprint(conduit::Node &output,
                                      const vtkm::cont::Field &field)
{
  std::string name = field.GetName();
  std::string path = "fields/" + name;
  bool assoc_points = vtkm::cont::Field::ASSOC_POINTS == field.GetAssociation();
  bool assoc_cells  = vtkm::cont::Field::ASSOC_CELL_SET == field.GetAssociation();
  //bool assoc_mesh  = vtkm::cont::Field::ASSOC_WHOLE_MESH == field.GetAssociation();
  if(!assoc_points && ! assoc_cells)
  {
    ASCENT_ERROR("Field must be associtated with cells or points\n");
  }
  std::string conduit_name;

  if(assoc_points) conduit_name = "vertex";
  else conduit_name = "element";
  
  output[path + "/association"] = conduit_name; 
  output[path + "/topology"] = "topo"; 

  vtkm::cont::DynamicArrayHandle dyn_handle = field.GetData(); 
  //
  // this can be literally anything. Lets do some exhaustive casting
  //
  if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Float32>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Float32>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Float64>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Float64>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Int8>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int8>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Int32>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int32>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Int64>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int64>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ASCENT_ERROR("Conduit int64 and vtkm::Int64 are different. Cannot convert vtkm::Int64\n");
    //output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::UInt32>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::UInt32>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::UInt8>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::UInt8>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues()); 
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,3>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,3>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else if(dyn_handle.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,2>>()))
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,2>>; 
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle);
  }
  else
  {
    field.PrintSummary(std::cerr);
    ASCENT_ERROR("Field type unsupported for conversion");
  }
}

void 
VTKHDataAdapter::VTKmToBlueprintDataSet(const vtkm::cont::DataSet *dset,
                                        conduit::Node &node)
{
  //
  // with vtkm, we have no idea what the type is of anything inside
  // dataset, so we have to ask all fields, cell sets anc coordinate systems.
  //
  const int default_cell_set = 0; 

  VTKmTopologyToBlueprint(node, *dset);

  const vtkm::Id num_fields = dset->GetNumberOfFields();
  for(vtkm::Id i = 0; i < num_fields; ++i)
  {
    vtkm::cont::Field field = dset->GetField(i);
    VTKmFieldToBlueprint(node, field);
  }
}

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



