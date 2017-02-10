//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
/// file: strawman_vtkm_pipeline.cpp
///
//-----------------------------------------------------------------------------
#include <strawman_vtkm_renderer.hpp>
#include "strawman_vtkm_pipeline_backend.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>

// thirdparty includes

// VTKm includes
#define VTKM_USE_DOUBLE_PRECISION
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderRectilinear.h>
#include <vtkm/rendering/Actor.h>

#ifdef VTKM_CUDA
#include <vtkm/cont/cuda/ChooseCudaDevice.h>
#endif

// other strawman includes
#include <strawman_block_timer.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin strawman:: --
//-----------------------------------------------------------------------------
namespace strawman
{
//-----------------------------------------------------------------------------
// -- VTKm typedefs for convenience and sanity
//-----------------------------------------------------------------------------
typedef vtkm::cont::DataSet                vtkmDataSet;
typedef vtkm::rendering::Actor             vtkmActor;


struct VTKMPipeline::Plot
{
    std::string        m_var_name;
    std::string        m_cell_set_name;
    bool               m_drawn;
    bool               m_hidden;
    vtkmDataSet       *m_data_set;     //typedefs are in renderer TODO: move to typedefs file
    vtkmActor         *m_plot;
    Node               m_render_options;
};


//-----------------------------------------------------------------------------
// VTKMPipeine::DataAdapter public methods
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKMPipeline::DataAdapter::BlueprintToVTKmDataSet
    (const Node &node, 
     const std::string &field_name)
{   
    STRAWMAN_BLOCK_TIMER(PIPELINE_GET_DATA);
    
    vtkm::cont::DataSet * result = NULL;
    // Follow var_name -> field -> topology -> coordset
    if(!node["fields"].has_child(field_name))
    {
        STRAWMAN_ERROR("Invalid field name " << field_name);
    }
    // as long as the field w/ field_name exists, and mesh blueprint verify 
    // is true, we access data without fear.
    
    const Node &n_field  = node["fields"][field_name];
    
    string topo_name     = n_field["topology"].as_string();
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
        STRAWMAN_ERROR("Unsupported topology/type:" << mesh_type);
    }
    
    // add var
    AddVariableField(field_name,
                     n_field,
                     topo_name,
                     neles,
                     nverts,
                     result);
   
    return result;
}


//-----------------------------------------------------------------------------
class ExplicitArrayHelper
{
public:
// Helper function to create explicit coordiante arrays for vtkm data sets
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
        STRAWMAN_ERROR("Unsupported element shape " << shape_type);
    }

    if(conn_size < indices) 
        STRAWMAN_ERROR("Connectivity array size " <<conn_size << " must be at least size " << indices);
    if(conn_size % indices != 0) 
        STRAWMAN_ERROR("Connectivity array size " <<conn_size << " be evenly divided by indices size" << indices);

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
#ifdef STRAWMAN_USE_OPENMP
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
VTKMPipeline::DataAdapter::UniformBlueprintToVTKmDataSet
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
    int dims_k = 0;

    // check for 3d
    if(n_dims.has_path("k"))
    {
        dims_k = n_dims["k"].to_int();
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
    
    vtkm::cont::CellSetStructured<3> cell_set(topo_name.c_str());
    cell_set.SetPointDimensions(dims);
    result->AddCellSet(cell_set);

    neles =  (dims_i - 1) * (dims_j - 1);
    if(dims_k > 0)
    {
        neles *= (dims_k - 1);
    }
    
    nverts =  dims_i * dims_j;
    if(dims_k > 0)
    {
        nverts *= dims_k;
    }

    return result;
}


//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKMPipeline::DataAdapter::RectilinearBlueprintToVTKmDataSet
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

    //The VTKM_USE_DOUBLE_PRECISION define make FloatDefault doubles
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
VTKMPipeline::DataAdapter::StructuredBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name 
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    STRAWMAN_ERROR("Blueprint Structured Mesh to VTKm DataSet Not Implemented");
    return NULL;
}



//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKMPipeline::DataAdapter::UnstructuredBlueprintToVTKmDataSet
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
    
    vtkm::cont::CellSetExplicit<> cell_set(nverts,
                                           topo_name.c_str());

    cell_set.Fill(shapes, num_indices, connectivity);
    
    result->AddCellSet(cell_set);
    
    STRAWMAN_INFO("neles "  << neles);
    
    return result;
}

//-----------------------------------------------------------------------------

void
VTKMPipeline::DataAdapter::AddVariableField
    (const std::string &field_name,
     const Node &n_field,
     const std::string &topo_name,
     int neles,
     int nverts,
     vtkm::cont::DataSet *dset)
{
    STRAWMAN_INFO("nverts "  << nverts);
    STRAWMAN_INFO("neles "  << neles);
    
    
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
        STRAWMAN_ERROR("VTKm exception:" << error.GetMessage());
    }

}





//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// VTKMPipeline Methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

VTKMPipeline::VTKMPipelineBackend()
{
  STRAWMAN_BLOCK_TIMER(CONSTRUCTOR)
}

//-----------------------------------------------------------------------------

VTKMPipeline::~VTKMPipelineBackend()
{
    delete m_renderer;
    Cleanup();
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main pipeline interface methods, which are by the strawman interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

void
VTKMPipeline::Initialize(const conduit::Node &options)
{
#if PARALLEL
    if(!options.has_path("mpi_comm"))
    {
        CONDUIT_ERROR("Missing Strawman::Open options missing MPI communicator (mpi_comm)");
    }

    int mpi_handle = options["mpi_comm"].value();
    MPI_Comm comm = MPI_Comm_f2c(mpi_handle);
    m_renderer = new Renderer<DEVICE_ADAPTOR>(comm);
#ifdef VTKM_CUDA
    //
    //  If we are using cuda, figure out how many devices we have and
    //  assign a GPU based on rank.
    //
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0 && device_count <= 256)
    {
        int rank;  
        MPI_Comm_rank(comm,&rank);
        int rank_device = rank % device_count;
        err = cudaSetDevice(rank_device);
        if(err != cudaSuccess)
        {
            STRAWMAN_ERROR("Failed to set GPU " 
                           <<rank_device
                           <<" out of "<<device_count
                           <<" GPUs. Make sure there"
                           <<" are an equal amount of"
                           <<" MPI ranks/gpus per node.");
        }
        else
        {

            char proc_name[100];
            int length=0;
            MPI_Get_processor_name(proc_name, &length);

        }
        cuda_device  = rank_device;
    }
    else
    {
        STRAWMAN_ERROR("VTKm GPUs is enabled but none found");
    }
#endif

#else
    m_renderer = new Renderer<DEVICE_ADAPTOR>;

#endif
    
}


//-----------------------------------------------------------------------------

void
VTKMPipeline::Cleanup()
{
   
}

//-----------------------------------------------------------------------------

void
VTKMPipeline::Publish(const conduit::Node &data)
{ 
    //
    // Delete the old plots and data sets
    //
    for(int i = 0; i < m_plots.size(); ++i)
    {
     delete m_plots[i].m_data_set;
     delete m_plots[i].m_plot; 
    }
    m_plots.clear();
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------

void
VTKMPipeline::Execute(const conduit::Node &actions)
{
    //
    // Loop over the actions
    //
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        if(!action.has_path("action"))
        {
            STRAWMAN_INFO("Warning : action malformed");
            action.print();
            std::cout<<"\n";
            continue;
        }
        STRAWMAN_INFO("Executing " << action["action"].as_string());
       
        if (action["action"].as_string() == "add_plot")
        {
            AddPlot(action);
        }
        else if (action["action"].as_string() == "add_filter")
        {
            STRAWMAN_INFO("VTKm add_filter not implemented");
        }
        else if (action["action"].as_string() == "draw_plots")
        {
            DrawPlots();
        }
        else
        {
            STRAWMAN_INFO("Warning : unknown action "<<action["action"].as_string());
        }
   }
}
//-----------------------------------------------------------------------------

void
VTKMPipeline::AddPlot(const conduit::Node &action)
{
    const std::string field_name = action["field_name"].as_string();

    vtkm::rendering::ColorTable color_table("Spectral");
    //
    // Create the plot.
    //
    Plot plot;
    plot.m_var_name = field_name;
    plot.m_drawn = false;
    plot.m_hidden = false;
    plot.m_data_set = DataAdapter::BlueprintToVTKmDataSet(m_data,field_name);
    
    // we need the topo name ...
    const Node &n_field  = m_data["fields"][field_name];
    string topo_name     = n_field["topology"].as_string();
    
    plot.m_cell_set_name = topo_name;
    try
    {
        STRAWMAN_BLOCK_TIMER(PLOT)
        if(!plot.m_data_set->HasCellSet(plot.m_cell_set_name))
            STRAWMAN_ERROR("AddPlot: no cell set named "<<plot.m_cell_set_name);

        int cell_set_index = plot.m_data_set->GetCellSetIndex(plot.m_cell_set_name);
        plot.m_plot = new vtkmActor(plot.m_data_set->GetCellSet(cell_set_index),
                                    plot.m_data_set->GetCoordinateSystem(),
                                    plot.m_data_set->GetField(field_name),
                                    color_table);
        if(action.has_path("render_options"))
        {
            plot.m_render_options = action.fetch("render_options"); 
        }
        else 
        {
            plot.m_render_options = conduit::Node(); 
        }
    }
    catch (vtkm::cont::Error error) 
    {
        STRAWMAN_ERROR("AddPlot Got the unexpected error: " << error.GetMessage() << std::endl);
    }
    m_plots.push_back(plot);
    
}

//-----------------------------------------------------------------------------

void 
VTKMPipeline::DrawPlots()
{
    for (int i = 0; i < m_plots.size(); ++i)
    {
        if(!m_plots[i].m_hidden)
        {
            RenderPlot(i, m_plots[i].m_render_options);
            m_plots[i].m_drawn = true;
        }
        else m_plots[i].m_drawn = false;
    }
}

//-----------------------------------------------------------------------------

void
VTKMPipeline::RenderPlot(const int plot_id,
                                                const conduit::Node &render_options)
{ 
    render_options.print();
    
    STRAWMAN_BLOCK_TIMER(RENDER_PLOTS);

    //
    // Extract the save image attributes.
    //

    int image_width  = 1024;
    int image_height = 1024;
    
    if(render_options.has_path("width"))
    {
        image_width = render_options["width"].to_int();
    }
    if(render_options.has_path("height"))
    {
        image_height = render_options["height"].to_int();
    }

    RendererType m_render_mode;
    //
    // Determine the render mode, default is opengl.
    //

    if(render_options.has_path("renderer"))
    {
        if(render_options["renderer"].as_string() == "volume")
        {
            m_render_mode = VOLUME;
        }
        else if(render_options["renderer"].as_string() == "raytracer")
        {
            m_render_mode = RAYTRACER;
        }
        else
        {
            STRAWMAN_INFO( "VTK-m Pipeline: unknown renderer "
                           << render_options["Renderer"].as_string() << endl
                           << "Defaulting to ray tracer");
            m_render_mode = RAYTRACER;
        }
    }
    else
    {
        m_render_mode = RAYTRACER;
    }
    
    const char *image_file_name = NULL;
    //
    // If a file name is provided, then save the image, otherwise start a web server
    //
    if(render_options.has_path("file_name"))
    {
       image_file_name = render_options["file_name"].as_char8_str();
    }
    else 
    {
        conduit::Node options;
        options["web/stream"] = "true";
        m_renderer->SetOptions(options);
    }
    
    //
    //    Check for camera attributes
    //
    if(render_options.has_path("camera")) 
    {
        m_renderer->SetCamera(render_options.fetch("camera"));
    }
    
    //
    // Check for Color Map
    //
    if(render_options.has_path("color_map"))
    {
        m_renderer->SetTransferFunction(render_options.fetch("color_map"));
    }
    int dims = 3;
    
    m_renderer->Render(m_plots[plot_id].m_plot,
                       image_height,
                       image_width,
                       m_render_mode,
                       dims,
                       image_file_name);
}

};
//-----------------------------------------------------------------------------
// -- end strawman:: --
//-----------------------------------------------------------------------------



