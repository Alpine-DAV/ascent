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
/// file: strawman_eavl_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "strawman_eavl_pipeline.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// eavl includes

// opengl (usually via osmesa)
//#include <GL/gl_mangle.h>

#include <eavl.h>
#include <eavlConfig.h>
#include <eavlCUDA.h>
#include <eavlDataSet.h>
#include <eavlArray.h>
#include <eavlExecutor.h>
#include <eavlCellSetExplicit.h>
#include <eavlCellSetAllStructured.h>
#include <eavlCoordinates.h>
#include <eavlLogicalStructureRegular.h>

#include <eavlPlot.h>
#include <eavl2DWindow.h>
#include <eavl3DWindow.h>
#include <eavlScene.h>

#include <eavlRenderSurfaceOSMesa.h>
#include <eavlSceneRendererGL.h>
#include <eavlSceneRendererRT.h>
#include <eavlSceneRendererSimpleGL.h>
#include <eavlSceneRendererSimplePVR.h>
#include <eavlSceneRendererSimpleVR.h>
#include <eavlWorldAnnotatorGL.h>
#include <eavlWorldAnnotatorPS.h>
#include <eavlRenderSurfacePS.h>

#include <eavlTransferFunction.h>
#include <eavlMatrix4x4.h>
#include <eavlVector3.h>
#include <eavlPoint3.h>

//--- eavl filters
#include <eavlCellToNodeRecenterMutator.h>
#include <eavlExternalFaceMutator.h>
#include <eavlThresholdMutator.h>
#include <eavlIsosurfaceFilter.h>
#include <eavlBoxMutator.h>


// mpi related includes
#ifdef PARALLEL
#include <mpi.h>
//----iceT includes 
#include <strawman_icet_compositor.hpp>
// -- conduit mpi
#include <conduit_relay_mpi.hpp>
#endif


// other strawman includes
#include <strawman_block_timer.hpp>
#include <strawman_png_encoder.hpp>
#include <strawman_web_interface.hpp>

using namespace conduit;

// def as 1 for more debug info
#define DEBUG   0

//-----------------------------------------------------------------------------
// -- begin strawman:: --
//-----------------------------------------------------------------------------
namespace strawman
{


//-----------------------------------------------------------------------------
struct EAVLPipeline::Plot
{
    std::string        m_var_name;
    std::string        m_cell_set_name;
    bool               m_drawn;
    bool               m_hidden;
    eavlDataSet       *m_eavl_dataset;
    eavlPlot          *m_eavl_plot;
    Node               m_render_options;
};


//-----------------------------------------------------------------------------
//
// TODO: should Visibility be a class?
//
//-----------------------------------------------------------------------------
struct Visibility
{
    int   m_rank;
    float m_minz;
};

//-----------------------------------------------------------------------------
int
CompareVisibility(const void *a, const void *b)
{
  if((*(Visibility*)a).m_minz <  (*(Visibility*)b).m_minz)  return -1;
  if((*(Visibility*)a).m_minz == (*(Visibility*)b).m_minz) return 0;
  if((*(Visibility*)a).m_minz >  (*(Visibility*)b).m_minz)  return 1;
  // TODO: default return?
  return -1;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Private Class that Handles Blueprint to EAVL Data Transforms
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class EAVLPipeline::DataAdapter
{
public:
    // convert blueprint data to an eavlDataSet
   static eavlDataSet     *BlueprintToEAVLDataSet(conduit::Node &n,
                                                  const std::string &field_name);


private:
    // helpers for specific conversion cases
    static eavlDataSet    *UniformBlueprintToEAVLDataSet(const std::string &coords_name,
                                                         const conduit::Node &n_coords,
                                                         const std::string &topo_name,
                                                         const conduit::Node &n_topo,
                                                         int &nele,
                                                         int &nverts);

    static eavlDataSet    *RectilinearBlueprintToEAVLDataSet(const std::string &coords_name,
                                                             const conduit::Node &n_coords,
                                                             const std::string &topo_name,
                                                             const conduit::Node &n_topo,
                                                             int &nele,
                                                             int &nverts);

    static eavlDataSet    *StructuredBlueprintToEAVLDataSet(const std::string &coords_name,
                                                            const conduit::Node &n_coords,
                                                            const std::string &topo_name,
                                                            const conduit::Node &n_topo,
                                                            int &nele,
                                                            int &nverts);

    static eavlDataSet    *UnstructuredBlueprintToEAVLDataSet(const std::string &coords_name,
                                                              const conduit::Node &n_coords,
                                                              const std::string &topo_name,
                                                              const conduit::Node &n_topo,
                                                              int &nele,
                                                              int &nverts);

    // helper for adding field data
    static void            AddVariableField(const std::string &field_name,
                                            const conduit::Node &n_field,
                                            const std::string &topo_name,
                                            int neles,
                                            int nverts,
                                            eavlDataSet *dset);
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Internal Class that Handles Rendering via EAVL
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class EAVLPipeline::Renderer
{
public:
      enum RenderMode { OPENGL,
                        VOLUME,
                        RAYTRACER };

      Renderer();

#ifdef PARALLEL
      Renderer(MPI_Comm mpi_comm);
#endif

      ~Renderer();
  
      void SetOptions(const Node &options);

      void SetTransferFunctionParams(Node &transfer_function);
      void SetCameraParams(Node &camera);
      
      void AddPlot(eavlPlot *plot);
      void SetData(Node *data_ptr);
  
      void ClearScene();

      void Render(eavlPlot *plot,
                  int image_width,
                  int image_height,
                  RenderMode mode,
                  int dims,
                  const char *image_file_name);
 
      void WebSocketPush(PNGEncoder &png);
      void WebSocketPush(const std::string &img_file_path);
  
  
private:

//-----------------------------------------------------------------------------
// private structs // classes
//-----------------------------------------------------------------------------

  struct RenderParams
  {
      public:
          int          m_width;
          int          m_height;
          RenderMode   m_mode;
          int          m_plot_dims;
    
          RenderParams()
          : m_width(-1),
            m_height(-1),
            m_mode(OPENGL),
            m_plot_dims(-1)
          {}
  };

//-----------------------------------------------------------------------------
// private methods
//-----------------------------------------------------------------------------
    void Defaults();
    void Cleanup();
    void ResetViewPlanes();
    void InitRendering(int plot_dims);
    
    void SetupTransferFunction(eavlTransferFunction *transfer_func);
    eavlColorTable  SetupColorTable();
    void SetupCamera();
    void CreateDefaultView(eavlPlot*, int);
//-----------------------------------------------------------------------------
// private methods for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL
    void  CheckIceTError();
    int  *FindVisibilityOrdering(eavlPlot *plot);
    void  SetParallelPlotExtents(eavlPlot * plot);
#endif
  

//-----------------------------------------------------------------------------
// private data members
//-----------------------------------------------------------------------------

    eavlWindow         *m_window;
    eavlRenderSurface  *m_surface;
    eavlSceneRenderer  *m_renderer;
    eavlScene          *m_scene;
    eavlWorldAnnotator *m_annotator;
    eavlColor           m_bg_color;
  
    RenderMode          m_render_mode;
    RenderParams        m_last_render_params;
  
    Node               *m_transfer_function;
    Node               *m_camera;
  
    Node               *m_data;

    // always keep rank, even for serial
    int                 m_rank;
    
    bool                m_web_stream_enabled; // CDH: move to pipeline ?
    
    WebInterface        m_web_interface;
    PNGEncoder          m_png_data;

//-----------------------------------------------------------------------------
// private vars for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL
    MPI_Comm            m_mpi_comm;
    
    IceTCompositor      m_icet;
    
    bool                m_image_subset_enabled;
    int                 m_mpi_size;
#endif 


};


//-----------------------------------------------------------------------------
// EAVLPipeine::DataAdapter public methods
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
eavlDataSet *
EAVLPipeline::DataAdapter::BlueprintToEAVLDataSet(Node &node, 
                                                  const std::string &field_name)
{   
    STRAWMAN_BLOCK_TIMER(PIPELINE_GET_DATA);
    
    eavlDataSet *result = NULL;
    
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
        result = UniformBlueprintToEAVLDataSet(coords_name,
                                               n_coords,
                                               topo_name,
                                               n_topo,
                                               neles,
                                               nverts);
    }
    else if(mesh_type == "rectilinear")
    {
        result = RectilinearBlueprintToEAVLDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts);
        
    }
    else if(mesh_type == "structured")
    {
        result =  StructuredBlueprintToEAVLDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts);
    }
    else if( mesh_type ==  "unstructured")
    {
        result =  UnstructuredBlueprintToEAVLDataSet(coords_name,
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
eavlDataSet *
EAVLPipeline::DataAdapter::UniformBlueprintToEAVLDataSet(const std::string &coords_name,
                                                         const conduit::Node &n_coords,
                                                         const std::string &topo_name,
                                                         const conduit::Node &n_topo,
                                                         int &neles,
                                                         int &nverts)
{
    // TODO: lets resolve this
    STRAWMAN_ERROR("Blueprint Uniform Mesh to EAVL DataSet Not Implemented");
    return NULL;
}


//-----------------------------------------------------------------------------
eavlDataSet *
EAVLPipeline::DataAdapter::RectilinearBlueprintToEAVLDataSet(const std::string &coords_name,
                                                             const conduit::Node &n_coords,
                                                             const std::string &topo_name,
                                                             const conduit::Node &n_topo,
                                                             int &neles,
                                                             int &nverts)
{
    eavlDataSet *result = new eavlDataSet();

    int dims = 2;
    if(n_coords.has_path("values/z"))
    {
        dims = 3;
    }

    const Node &n_coords_x = n_coords["values/x"];
    const Node &n_coords_y = n_coords["values/y"];

    // TODO: assumes double
    
    const double *x_values = n_coords_x.as_double_ptr();
    const double *y_values = n_coords_y.as_double_ptr();
    const double *z_values = NULL;
    
    if(dims == 3)
    {
        z_values = n_coords["values/z"].as_double_ptr();
    }
    
    // Set the number of points 
    int nx = n_coords_x.dtype().number_of_elements();
    int ny = n_coords_y.dtype().number_of_elements();
    int nz = 0;
    
    if(dims == 3)
    { 
        nz  = n_coords["values/z"].dtype().number_of_elements();
    }
        
    if(dims == 2)
    {
        neles  = (nx - 1) * (ny - 1);
        nverts = nx * ny;
    }
    else // if dims == 3
    {
        neles  = (nx - 1) * (ny - 1) * (nz - 1);
        nverts = nx * ny * nz;
    }
    
    result->SetNumPoints(nverts);
    
    // Set the logical structure 
    eavlRegularStructure reg;
    if(dims == 2)
    {
        reg.SetNodeDimension2D(nx, ny);
    }
    else 
    {
        reg.SetNodeDimension3D(nx, ny, nz);
    }
    
    eavlLogicalStructure *logical_st =
        new eavlLogicalStructureRegular(reg.dimension, reg);

    result->SetLogicalStructure(logical_st);

    // Create the coordinate axes.
    // Note: Changed size to number of elements since the size would
    //       be nx for rectilinear, but npts for curvilinear. We assume
    //       that the correct number is the size of the array.
    
    eavlDoubleArray *x = new eavlDoubleArray(eavlArray::HOST,
                                             const_cast<double*>(x_values),
                                             "x",
                                             1,
                                             n_coords_x.dtype().number_of_elements());

    eavlDoubleArray *y = new eavlDoubleArray(eavlArray::HOST,
                                             const_cast<double*>(y_values),
                                             "y",
                                             1,
                                             n_coords_y.dtype().number_of_elements());
   

    eavlDoubleArray *z = NULL;
    
    if(dims > 2)
    {
         z = new eavlDoubleArray(eavlArray::HOST,
                                 const_cast<double*>(z_values),
                                 "z",
                                 1,
                                 n_coords["values/z"].dtype().number_of_elements());
    }


    result->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    result->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));
    
    if(dims == 3)
    {
        result->AddField(new eavlField(1, z, eavlField::ASSOC_LOGICALDIM, 2));
    }

    
    // Set the coordinates
    eavlCoordinates *coords = NULL;
    
    if(dims == 2)
    {
        coords = new eavlCoordinatesCartesian(logical_st,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
    }
    else  // if dims == 3
    {
        coords = new eavlCoordinatesCartesian(logical_st,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);
    }


    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    
    if(dims == 3)
    {
        coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    }
    
    result->AddCoordinateSystem(coords);

    // Create a cell set implicitly covering the entire regular structure
    eavlCellSet *cell_set = new eavlCellSetAllStructured("cells", reg);
    result->AddCellSet(cell_set);

    return result;
}



//-----------------------------------------------------------------------------
eavlDataSet *
EAVLPipeline::DataAdapter::StructuredBlueprintToEAVLDataSet(const std::string &coords_name,
                                                            const conduit::Node &n_coords,
                                                            const std::string &topo_name,
                                                            const conduit::Node &n_topo,
                                                            int &neles,
                                                            int &nverts)
{
    STRAWMAN_ERROR("Blueprint Structured Mesh to EAVL DataSet Not Implemented");
    return NULL;
}


//-----------------------------------------------------------------------------
eavlDataSet *
EAVLPipeline::DataAdapter::UnstructuredBlueprintToEAVLDataSet(const std::string &coords_name,
                                                              const conduit::Node &n_coords,
                                                              const std::string &topo_name,
                                                              const conduit::Node &n_topo,
                                                              int &neles,
                                                              int &nverts)
{
    eavlDataSet *result = new eavlDataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();
    
    // set the number of points
    result->SetNumPoints(nverts);

    // no logical structure
    eavlLogicalStructure *logical_st = NULL;

    // not sure how to zero copy?
    // create the coordinate axes
   
    int32 ndims = 2;
    
    // TODO: assumes doubles 
    const double *x_coords_data = n_coords["values/x"].as_double_ptr();
    const double *y_coords_data = n_coords["values/y"].as_double_ptr();
    const double *z_coords_data = NULL;
    
    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        z_coords_data = n_coords["values/z"].as_double_ptr();
    }
    
    eavlDoubleArray *x_coords = new eavlDoubleArray(eavlArray::HOST,
                                                    const_cast<double*>(x_coords_data),
                                                    "x",
                                                    1,
                                                    nverts);

    result->AddField(new eavlField(1, x_coords, eavlField::ASSOC_POINTS));

    eavlDoubleArray *y_coords = new eavlDoubleArray(eavlArray::HOST,
                                                    const_cast<double*>(y_coords_data),
                                                    "y",
                                                    1,
                                                    nverts);


    result->AddField(new eavlField(1, y_coords, eavlField::ASSOC_POINTS));



    if(ndims == 3)
    {
        eavlDoubleArray *z_coords = new eavlDoubleArray(eavlArray::HOST,
                                                        const_cast<double*>(z_coords_data),
                                                        "z",
                                                        1,
                                                        nverts);
    
        result->AddField(new eavlField(1, z_coords, eavlField::ASSOC_POINTS));
    }
    

    eavlCoordinates *coords = NULL;
    // set the coordinates
    if(ndims == 2)
    {
        coords = new eavlCoordinatesCartesian(logical_st,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
    }
    else
    {
        coords = new eavlCoordinatesCartesian(logical_st,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);
    }
    
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    
    if(ndims == 3)
    {
        coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    }
    
    result->AddCoordinateSystem(coords);
    
    string ele_shape = n_topo["elements/shape"].as_string();

    const Node &n_topo_ele_conn = n_topo["elements/connectivity"];
    
    if(ele_shape == "hex")
    {
        neles = n_topo_ele_conn.dtype().number_of_elements() / 8;
        // create a topologically 3D cell set with hexs
        eavlCellSetExplicit *cells = new eavlCellSetExplicit("cells", 3);
        eavlExplicitConnectivity conn;
        // TODO:assumes int was used
        const int *ele_idx_ptr = n_topo_ele_conn.value();
        for(int i=0; i < neles; i++)
        {
            conn.AddElement(EAVL_HEX, 8,  const_cast<int*>(ele_idx_ptr));
            ele_idx_ptr+=8;
        }
        cells->SetCellNodeConnectivity(conn);
        result->AddCellSet(cells);
    }
    else if(ele_shape == "quad")
    {
    
        neles = n_topo_ele_conn.dtype().number_of_elements() / 4;
        // create a topologically 2D cell set with quads
        eavlCellSetExplicit *cells = new eavlCellSetExplicit("cells", 2);
        eavlExplicitConnectivity conn;
        // assumes int was used
        const int *ele_idx_ptr = n_topo_ele_conn.value();
        for(int i=0; i < neles; i++)
        {
            conn.AddElement(EAVL_QUAD, 4, const_cast<int*>(ele_idx_ptr));
            ele_idx_ptr+=4;
        }
        cells->SetCellNodeConnectivity(conn);
        result->AddCellSet(cells);
    }
    else
    {
        STRAWMAN_ERROR(ele_shape << " element shape not supported.");
    }

    return result;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::DataAdapter::AddVariableField(const std::string &field_name,
                                            const conduit::Node &n_field,
                                            const std::string &topo_name,
                                            int neles,
                                            int nverts,
                                            eavlDataSet *dset)
{   
    // TODO: assumes double
    const double *values_ptr = n_field["values"].as_float64_ptr();
    string assoc             = n_field["association"].as_string();
    if ( assoc == "vertex")
    {
        eavlDoubleArray *field = new eavlDoubleArray(eavlArray::HOST, 
                                                     const_cast<double*>(values_ptr),
                                                     field_name.c_str(),
                                                     1,
                                                     nverts);
                                    
        dset->AddField(new eavlField(0,
                                     field,
                                     eavlField::ASSOC_POINTS,
                                     // having var name here seems strange, but
                                     // its necessary ?
                                     field_name.c_str()));
    }
    else if (assoc == "element")
    {
        eavlDoubleArray *field = new eavlDoubleArray(eavlArray::HOST,
                                                     const_cast<double*>(values_ptr),
                                                     field_name.c_str(),
                                                     1,
                                                     neles);

        dset->AddField(new eavlField(0,
                                     field,
                                     eavlField::ASSOC_CELL_SET,
                                     "cells"));
    }
    else
    {
        STRAWMAN_ERROR("Unsupported field association: " << assoc);
    }
}



//-----------------------------------------------------------------------------
// EAVLPipeline::Renderer public methods
//-----------------------------------------------------------------------------
EAVLPipeline::Renderer::Renderer()
: m_window(NULL),
  m_surface(NULL),
  m_scene(NULL),
  m_renderer(NULL),
  m_annotator(NULL),
  m_rank(0),
  m_web_stream_enabled(false)
{
    Defaults();
    m_camera = NULL;
    m_transfer_function = NULL;
}


//-----------------------------------------------------------------------------
#ifdef PARALLEL
//-----------------------------------------------------------------------------
EAVLPipeline::Renderer::Renderer(MPI_Comm mpi_comm)
: m_window(NULL),
    m_surface(NULL),
    m_scene(NULL),
    m_renderer(NULL),
    m_annotator(NULL),
    m_rank(0),
    m_web_stream_enabled(false),
    m_mpi_comm(mpi_comm),
    m_image_subset_enabled(false)
{
    Defaults();
    m_camera = NULL;
    m_transfer_function = NULL;
    m_icet.Init(m_mpi_comm);

    MPI_Comm_rank(m_mpi_comm, &m_rank);
    MPI_Comm_size(m_mpi_comm, &m_mpi_size);

}
#endif


//-----------------------------------------------------------------------------
EAVLPipeline::Renderer::~Renderer()
{
    STRAWMAN_BLOCK_TIMER(RENDERER_ON_DESTROY);
    
    Cleanup();

#ifdef PARALLEL
    m_icet.Cleanup();
#endif
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetOptions(const Node &options)
{
    // this is where we grab env vars
        
    if(options.has_path("web/stream") && 
       options["web/stream"].as_string() == "true")
    {
        m_web_stream_enabled = true;
    }
    
    
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetData(Node *data_node_ptr)
{
     m_data = data_node_ptr;
}
  
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetTransferFunctionParams(Node &transfer_function_params)
{
    m_transfer_function = &transfer_function_params;
}
  
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetCameraParams(Node &camera_params)
{
    m_camera = &camera_params;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::AddPlot(eavlPlot *plot)
{
    m_scene->plots.push_back(plot);
}
  
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::ClearScene()
{
    //
    // now for eavl ( we shouldn't have to do this...)
    // we can't just delete the scene, since m_window has a pointer
    // to it. We would basically have to re-init all rendering.
    // TODO: add clear scene in eavl
    // 
    if(m_scene != NULL)
    {
        int num_plots = m_scene->plots.size();
        
        for(int i = 0; i < num_plots; i++)
        {
            delete m_scene->plots.at(i);
        }
        
        m_scene->plots.clear();
    }
}
 
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::WebSocketPush(PNGEncoder &png)
{
    // no op if web streaming isn't enabled
    if( !m_web_stream_enabled )
    {
        return;
    }
    
    

    // we want to send the number of domains as part of the status msg
    // collect that from all procs
    // TODO: support domain overloading 
    int ndomains = 1;

#if PARALLEL
    Node n_src, n_rcv;
    n_src = ndomains; 
    relay::mpi::all_reduce(n_src,
                           n_rcv,
                           MPI_INT,
                           MPI_SUM,
                           m_mpi_comm);
    ndomains = n_rcv.value();
#endif
    
    // the rest only needs to happen on the root proc
    if( m_rank != 0)
    {
        return;
    }
    STRAWMAN_INFO("WebSocketPush");
    
    Node msg;
    msg["type"] = "status";
    msg["data"] = m_data->fetch("state");
    msg["data"].remove("domain");
    msg["data/ndomains"] = ndomains;
    msg.print();
    
    m_web_interface.PushMessage(msg);
    m_web_interface.PushImage(png);
 }


//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::WebSocketPush(const std::string &img_file_path)
{
    // no op if web streaming isn't enabled
    if( !m_web_stream_enabled )
    {
        return;
    }

    // we want to send the number of domains as part of the status msg
    // collect that from all procs
    // TODO: Support domain overload ( M domains per MPI task)
    int ndomains = 1;

#if PARALLEL
    Node n_src, n_rcv;
    n_src = ndomains; 
    relay::mpi::all_reduce(n_src,
                           n_rcv,
                           MPI_INT,
                           MPI_SUM,
                           m_mpi_comm);
    ndomains = n_rcv.value();
#endif
    
    // the rest only needs to happen on the root proc
    if( m_rank != 0)
    {
        return;
    }
    
    STRAWMAN_INFO("WebSocketPush");
    
    Node msg;
    msg["type"] = "status";
    msg["data"] = m_data->fetch("state");
    msg["data"].remove("domain");
    msg["data/ndomains"] = ndomains;
    msg.print();
    
    m_web_interface.PushMessage(msg);
    
    std::string img_file_path_full(img_file_path);
    img_file_path_full = img_file_path_full + ".png";

    m_web_interface.PushImage(img_file_path_full);
 }


//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::Render(eavlPlot *plot,
                               int image_width,
                               int image_height,
                               RenderMode mode,
                               int dims,
                               const char *image_file_name)
{
    try
    {
        PNGEncoder png;
        //
        // Do some check to see if we need
        // to re-init rendering
        //

        m_render_mode = mode;
        
        bool render_dirty = false;
        bool screen_dirty = false;
        
        if(m_render_mode != m_last_render_params.m_mode)
        {
            render_dirty = true;
        }
        
        if(dims != m_last_render_params.m_plot_dims)
        {
            render_dirty = true;
        }
        
        if(image_width  != m_last_render_params.m_width || 
           image_height != m_last_render_params.m_height)
        {
            screen_dirty = true;
        }

        m_last_render_params.m_mode      = m_render_mode;
        m_last_render_params.m_plot_dims = dims;
        m_last_render_params.m_width     = image_width;
        m_last_render_params.m_height    = image_height;


        if(render_dirty)
        {
            InitRendering(dims);
            m_window->Initialize();
        }
        
        if(screen_dirty)
        {
            m_window->Resize(image_width, image_height);
        }
        
        m_scene->plots.push_back(plot);
          
               
        //
        // Check for transfer function
        //
        if(m_render_mode == VOLUME)
        {
            if(m_transfer_function != NULL)
            {
                eavlTransferFunction *tf = 
                  static_cast<eavlSceneRendererSimplePVR*>(m_renderer)->GetTransferFunction();
                tf->Clear();
                SetupTransferFunction(tf);
                static_cast<eavlSceneRendererSimplePVR*>(m_renderer)->SampleTransferFunction();
                //
                // set the color bar
                //
                m_scene->plots[0]->SetColorTable(*tf->GetRGBColorTable());
              }
          }
        else
        {
            if(m_transfer_function != NULL) plot->SetColorTable(SetupColorTable());
        }
          
        CreateDefaultView(m_scene->plots[0], dims);
#ifdef PARALLEL
        //
        //  We need to turn off the background for the
        //  parellel volume render BEFORE the scene
        //  is painted. 
        //
        
        int *vis_order = NULL;
        if(m_render_mode == VOLUME)
        {
            eavlSceneRendererSimplePVR *vr = 
                static_cast<eavlSceneRendererSimplePVR*>(m_renderer);
            vr->SetTransparentBG(true);

            //
            // Calculate visibility ordering AFTER 
            // the camera parameters have been set
            // IceT uses this list to composite the images
            
            //
            // TODO: This relies on plot 0
            //
            vis_order = FindVisibilityOrdering(m_scene->plots[0]);
        }
#endif
        //
        // Check to see if we have camera params
        //
        if(m_camera != NULL)
        {  
            SetupCamera();
        }
 //---------------------------------------------------------------------
        {// open block for RENDER_PAINT Timer
        //---------------------------------------------------------------------
            STRAWMAN_BLOCK_TIMER(RENDER_PAINT);

#ifdef PARALLEL
            m_window->DisableAnnotations(true);
            if(m_render_mode != OPENGL)
            {
                m_scene->Render(m_window);
            }  //GL renderer does not return pixels, only the surface
            else
            {
               m_window->Paint();
            }
#else
            m_window->DisableAnnotations(false);
            m_window->Paint();
#endif
        //---------------------------------------------------------------------
        } // close block for RENDER_PAINT Timer
        //---------------------------------------------------------------------
        
        //Save the image.
        const unsigned char *result_color_buffer = NULL;
#ifdef PARALLEL
        //---------------------------------------------------------------------
        {// open block for RENDER_COMPOSITE Timer
        //---------------------------------------------------------------------
            STRAWMAN_BLOCK_TIMER(RENDER_COMPOSITE);
            //
            // init IceT parallel image compositing
            //
            int view_port[4] = {0,
                                0,
                                image_width,
                                image_height};

            if(m_image_subset_enabled)
            { 
      
                if(m_render_mode == RAYTRACER)
                {
                    static_cast<eavlSceneRendererRT*>(m_renderer)->getImageSubsetDims(view_port);
                }
                if(m_render_mode == VOLUME)
                {
                    static_cast<eavlSceneRendererSimplePVR*>(m_renderer)->getImageSubsetDims(view_port);
                }
            }
        
            const unsigned char *input_color_buffer  = NULL;
            const float         *input_depth_buffer  = NULL;    
            
            if(m_render_mode != OPENGL)
            {
                // TODO ? "apparently" was a comment here ... 
                input_color_buffer = m_renderer->GetRGBAPixels();
                input_depth_buffer = m_renderer->GetDepthPixels();
            }
            else
            {
                input_color_buffer = ((eavlRenderSurfaceOSMesa*) m_surface)->GetRGBABuffer();
                input_depth_buffer = ((eavlRenderSurfaceOSMesa*) m_surface)->GetZBuffer();
            }
            
            
            if(m_render_mode != VOLUME)
            {   
                result_color_buffer = m_icet.Composite(image_width,
                                                       image_height,
                                                       input_color_buffer,
                                                       input_depth_buffer,
                                                       view_port,
                                                       m_bg_color.c);
            }
            else
            {    
                //
                // Volume rendering uses a visibility ordering 
                // by rank instead of a depth buffer
                //
                result_color_buffer = m_icet.Composite(image_width,
                                                       image_height,
                                                       input_color_buffer,
                                                       vis_order,
                                                       m_bg_color.c);
                // leak?
                free(vis_order);
            }

    
        //---------------------------------------------------------------------
        }// close block for RENDER_COMPOSITE Timer
        //---------------------------------------------------------------------
        
#else
        // TODO: for serial case, we need to get the rgba buffer
        // out of EAVL, so we can png encode, then base64 encode it
        // without reading the data from a file.
   
        
        if(m_render_mode != OPENGL)
        {
            result_color_buffer = m_renderer->GetRGBAPixels();
        }
        else
        {
            result_color_buffer = ((eavlRenderSurfaceOSMesa*) m_surface)->GetRGBABuffer();
        }
        
#endif
        //
        // If we have a file name, write to disk, otherwise stream
        //
        bool encode_image = true;
#ifdef PARALLEL
        if(m_rank != 0) encode_image = false;
#endif
        if(encode_image)
        {
            m_png_data.Encode(result_color_buffer,
                              image_width,
                              image_height);
        }
        if(image_file_name != NULL)
        {
            bool save_image = true;
#ifdef PARALLEL
            if(m_rank != 0) save_image = false;
#endif
            string ofname(image_file_name);
            ofname +=  ".png";
            if(save_image)
            {
                m_png_data.Save(ofname);
            }
        }
        else
        // png will be null if rank !=0, thats fine
            WebSocketPush(m_png_data);
    }// end try

    catch(const eavlException &e)
    {
        STRAWMAN_WARN("eavlException when Rendering " <<  e.GetErrorText());
    }
}
//-----------------------------------------------------------------------------
// imp EAVLPipeline::Renderer private methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::Defaults()
{
    m_bg_color.c[0] = .5f;
    m_bg_color.c[1] = .5f;
    m_bg_color.c[2] = .5f;
    m_bg_color.c[3] = 1.0f;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::Cleanup()
{
    if(m_window)
    {
        delete m_window;
        m_window = NULL;

    }
    
    if(m_surface)
    {
        delete m_surface;
        m_surface = NULL;
    }

    if(m_renderer)
    {
        delete m_renderer;
        m_renderer = NULL;
    }
    
    if(m_scene)
    {
        delete m_scene;
        m_scene = NULL;
    }

    if(m_annotator)
    {
        delete m_annotator;
        m_annotator = NULL;
    }
}


  
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::ResetViewPlanes()
{
    //
    // The squeezes the view planes against the data set
    // so we get maximum sampling for the volume renderer
    // Note: for 3D only, will have no effect on 2D plots
    //
    eavlView &view = m_window->view;
    view.SetupMatrices();
    eavlPoint3 mins(view.minextents[0],view.minextents[1],view.minextents[2]);
    eavlPoint3 maxs(view.maxextents[0],view.maxextents[1],view.maxextents[2]);
    double x[2], y[2], z[2];
    x[0] = view.minextents[0];
    y[0] = view.minextents[1];
    z[0] = view.minextents[2];
    x[1] = view.maxextents[0];
    y[1] = view.maxextents[1];
    z[1] = view.maxextents[2];
    float zmin, zmax;
    zmin = std::numeric_limits<float>::max();
    zmax = std::numeric_limits<float>::min();

    eavlPoint3 extentPoint;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            for(int k = 0; k < 2; k++)
            {
                extentPoint.x = x[i];
                extentPoint.y = y[j];
                extentPoint.z = z[k];
                extentPoint = view.V * extentPoint;
                zmin = std::min(zmin, -extentPoint.z);
                zmax = std::max(zmax, -extentPoint.z);
            }
        }
    }
    
    view.view3d.nearplane = zmin * 0.9f; 
    view.view3d.farplane =  zmax * 1.1f; 
    view.SetupMatrices(); 
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::InitRendering(int plot_dims)
{
    STRAWMAN_BLOCK_TIMER(RENDER_INIT);
    
    // start from scratch
    Cleanup();
    
    // 
    //   Note: There is a disconnect here between spacial dimensions and 
    //         topological dimension. Cellset/plot dims are topological.
    //

#ifdef PARALLEL
    if(plot_dims == 99 )
    {
        m_scene = new eavl2DScene();
    }
    else
    {
        m_scene = new eavl3DScene();
    }
    
    if(m_render_mode == OPENGL)
    {
        m_surface   = new eavlRenderSurfaceOSMesa();
        m_annotator = new eavlWorldAnnotatorGL();
    }
    else
    {
        m_surface   = new eavlRenderSurfacePS();
        m_annotator = new eavlWorldAnnotatorPS();
    }
#else

    if(plot_dims == 99)
    {
        m_scene = new eavl2DScene();
    }
    else //if(plot_dims == 3)
    {
        m_scene = new eavl3DScene();
    }
    
    m_surface  = new eavlRenderSurfaceOSMesa();
    m_annotator = new eavlWorldAnnotatorGL();
#endif
    if(m_scene == NULL)
    {
          STRAWMAN_ERROR("Error: m_scene not created. "
                        << plot_dims 
                        << "  dimensional plots not currently supported");
    }
    //
    // Create the appropriate renderer
    //      
    m_renderer = NULL;
    if(plot_dims == 3)
    {
        if(m_render_mode == VOLUME)
        {
            m_renderer = new eavlSceneRendererSimplePVR();
        }
        else if(m_render_mode == RAYTRACER)
        {   
            m_renderer = new eavlSceneRendererRT();
        }
        else if(m_render_mode == OPENGL)
        {
            m_renderer = new eavlSceneRendererSimpleGL();
        }
    }
    else if(plot_dims == 2)
    {
        if(m_render_mode == RAYTRACER)
        {
            m_renderer = new eavlSceneRendererRT();
        }
        else if(m_render_mode == OPENGL)
        {
            m_renderer = new eavlSceneRendererSimpleGL();
        }
    }
    
    if(m_renderer == NULL)
    {
        STRAWMAN_ERROR("eavlSceneRenderer was not created");
    }
    m_window = NULL; 

    if(plot_dims == 1)
    {
        m_window = new eavl2DWindow(m_bg_color, 
                                    m_surface,
                                    m_scene, 
                                    m_renderer,
                                    m_annotator);
    }

#ifdef PARALLEL
    else if(plot_dims == 3 || plot_dims == 2)
    {
        m_window = new eavl3DWindow(m_bg_color, 
                                    m_surface,
                                    m_scene, 
                                    m_renderer,
                                    m_annotator);
                                  
    //
    // Image subset mode. Each renderer only renders pixels
    // that contain the AABB of the dataset. Memory reduction, times savings.
    // Only supported in certain renderers
    //
    if(m_render_mode == RAYTRACER || m_render_mode == VOLUME)
    { 
        m_image_subset_enabled = true;
        if(m_render_mode == RAYTRACER)
        {
            static_cast<eavlSceneRendererRT*>(m_renderer)->enableImageSubset(true);
        }
    }
#else
    if(plot_dims == 3 || plot_dims == 2)
    {
        m_window = new eavl3DWindow(m_bg_color, 
                                    m_surface,
                                    m_scene, 
                                    m_renderer,
                                    m_annotator);
#endif
      if(m_window == NULL)
      {
          STRAWMAN_ERROR("eavlWindow was not created.");
      }
    
  }
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetupTransferFunction(eavlTransferFunction *transfer_func)
{
     
    std::string color_map_name = "";
    if(m_transfer_function->has_path("name"))
        color_map_name = (*m_transfer_function)["name"].as_string();

    
    if(!m_transfer_function->has_path("control_points"))
    {
        if(color_map_name == "") 
          STRAWMAN_ERROR("Error: a color map node was provided without a color map name or control points");
        transfer_func->SetByColorTableName(color_map_name);
        return;
    }
    
    transfer_func->SetByColorTableName(color_map_name);
    if(color_map_name == "") transfer_func->Clear();
    
    NodeIterator itr = m_transfer_function->fetch("control_points").children();
    
    while(itr.has_next())
    {
        Node &peg = itr.next();
        
        float64 position = peg["position"].as_float64();
        
        if(position > 1.0 || position < 0.0)
        {
              STRAWMAN_WARN("Cannot add transfer function peg point position "
                            << position 
                            << ". Must be a normalized scalar.");
        }
  
        //
        // Should we give a warning for bad color or alpha values?
        // It wont't cause a seg fault.
        //
        
        if (peg["type"].as_string() == "rgb")
        {
            float64 *color = peg["color"].as_float64_ptr();
            
            eavlColor ecolor(color[0], color[1], color[2]);
            
            transfer_func->AddRGBControlPoint(position, ecolor);
        }
        else if (peg["type"].as_string() == "alpha")
        {
            float64 alpha = peg["alpha"].to_float64();
            
            transfer_func->AddAlphaControlPoint(position, alpha);
        }
        else
        {
            STRAWMAN_WARN("Unknown peg point type "<<peg["type"].as_string());
        }
    }
}
//-----------------------------------------------------------------------------
eavlColorTable
EAVLPipeline::Renderer::SetupColorTable()
{
     
    std::string color_map_name = "";
    if(m_transfer_function->has_path("name"))
        color_map_name = (*m_transfer_function)["name"].as_string();
    eavlColorTable color_map(color_map_name);
    if(color_map_name == "") color_map.Clear();
    
    if(!m_transfer_function->has_path("control_points"))
    {
        if(color_map_name == "") 
          STRAWMAN_ERROR("Error: a color map node was provided without a color map name or control points");
        return color_map;
    }
    
    NodeIterator itr = m_transfer_function->fetch("control_points").children();
    
    while(itr.has_next())
    {
        Node &peg = itr.next();
        
        float64 position = peg["position"].as_float64();
        
        if(position > 1.0 || position < 0.0)
        {
              STRAWMAN_WARN("Cannot add transfer function peg point position "
                            << position 
                            << ". Must be a normalized scalar.");
        }
  
        //
        // Should we give a warning for bad color or alpha values?
        // It wont't cause a seg fault.
        //
        
        if (peg["type"].as_string() == "rgb")
        {
            float64 *color = peg["color"].as_float64_ptr();
            
            eavlColor ecolor(color[0], color[1], color[2]);
            
            color_map.AddControlPoint(position, ecolor);
        }
        else if (peg["type"].as_string() == "alpha")
        {
          // Do nothing. eavlColorTables don't have a notion of alpha
        }
        else
        {
            STRAWMAN_WARN("Unknown peg point type "<<peg["type"].as_string());
        }
    }

    return color_map;
}
  
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetupCamera()
{
    if(m_camera == NULL)
    {
        return;
    }
    
    Node & node = *m_camera;
    
    eavlView &view = m_window->view;
    
    if(view.viewtype == eavlView::EAVL_VIEW_3D)
    {
        //
        // Get the optional camera parameters
        //
        if(node.has_path("look_at"))
        {
            float64 *coords = node["look_at"].as_float64_ptr();
            view.view3d.at = eavlPoint3(coords[0], 
                                        coords[1], 
                                        coords[2]);
        }
        if(node.has_path("position"))
        {
              float64 *coords = node["position"].as_float64_ptr();
              view.view3d.from = eavlPoint3(coords[0], 
                                            coords[1], 
                                            coords[2]);
        }
        
        if(node.has_path("up"))
        {
            float64 *coords = node["up"].as_float64_ptr();
            view.view3d.up = eavlVector3(coords[0], 
                                         coords[1], 
                                         coords[2]);
            view.view3d.up.normalize();
        }
        
        if(node.has_path("fov"))
        {
            view.view3d.fov = node["fov"].to_float64();
        }

        if(node.has_path("xpan"))
        {
            view.view3d.xpan = node["xpan"].to_float64();
        }

        if(node.has_path("ypan"))
        {
            view.view3d.ypan = node["ypan"].to_float64();
        }

        if(node.has_path("zoom"))
        {
            view.view3d.zoom = node["zoom"].to_float64();
        }

        if(node.has_path("neaplane"))
        {
            view.view3d.nearplane = node["nearplane"].to_float64();
        }

        if(node.has_path("farplane"))
        {
            view.view3d.nearplane = node["farplane"].to_float64();
        }
        
        if(node.has_path("xrotate"))
        {
            float64 xradians = node["xrotate"].to_float64();
            eavlMatrix4x4 rot;
            rot.CreateRotateX(xradians);
            view.view3d.from = rot * view.view3d.from;
            ResetViewPlanes();
        }

        if(node.has_path("yrotate"))
        {
            float64 yradians = node["yrotate"].to_float64();
            eavlMatrix4x4 rot;
            rot.CreateRotateY(yradians);
            view.view3d.from = rot * view.view3d.from;
            ResetViewPlanes();
        }

        if(node.has_path("zrotate"))
        {
            float64 zradians = node["zrotate"].to_float64();
            eavlMatrix4x4 rot;
            rot.CreateRotateZ(zradians);
            view.view3d.from = rot * view.view3d.from;
            ResetViewPlanes();
        }
        view.SetupMatrices();
    }
}




//-----------------------------------------------------------------------------
// imp EAVLPipeline::Renderer private methods for MPI case
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::CreateDefaultView(eavlPlot *plot, int dims)
{
    eavlView &view = m_window->view;
    // Code from eavl
    view.minextents[0] = view.minextents[1] = view.minextents[2] = FLT_MAX;
    view.maxextents[0] = view.maxextents[1] = view.maxextents[2] = -FLT_MAX;

    for (int d=0; d<3; d++)
    {
        float64 vmin = plot->GetMinCoordExtentOrig(d);
        if(vmin < view.minextents[d])
            view.minextents[d] = vmin;
        float64 vmax = plot->GetMaxCoordExtentOrig(d);
        if(vmax > view.maxextents[d])
            view.maxextents[d] = vmax;
    }
    //  This is a hack to mimic the internal behavior of EAVL
    //  Rendering 2D plots using opengl has backface culling 
    //  in seemingly opposite directions for the 2d and 3d cases.
    float64 camera_direction = -1;
#ifdef PARALLEL
    SetParallelPlotExtents(plot);
    camera_direction = 1;
#endif

    if (view.minextents[0] > view.maxextents[0])
        view.minextents[0] = view.maxextents[0] = 0;
    if (view.minextents[1] > view.maxextents[1])
        view.minextents[1] = view.maxextents[1] = 0;
    if (view.minextents[2] > view.maxextents[2])
        view.minextents[2] = view.maxextents[2] = 0;

    view.size = sqrt(pow(view.maxextents[0]-view.minextents[0],2.) +
                     pow(view.maxextents[1]-view.minextents[1],2.) +
                     pow(view.maxextents[2]-view.minextents[2],2.));
    //
    // For simplicity, always setup a 3d view even if plot is 2d
    
    float64 ds_size = sqrt( (view.maxextents[0]-view.minextents[0])*(view.maxextents[0]-view.minextents[0]) +
                            (view.maxextents[1]-view.minextents[1])*(view.maxextents[1]-view.minextents[1]) +
                            (view.maxextents[2]-view.minextents[2])*(view.maxextents[2]-view.minextents[2]) );

    eavlPoint3 center = eavlPoint3((view.maxextents[0]+view.minextents[0]) / 2.,
                                   (view.maxextents[1]+view.minextents[1]) / 2.,
                                   (view.maxextents[2]+view.minextents[2]) / 2.);
    
    view.viewtype = eavlView::EAVL_VIEW_3D;
    view.view3d.perspective = true;
    view.view3d.xpan = 0;
    view.view3d.ypan = 0;
    view.view3d.zoom = 1.0;
    view.view3d.at = center;
    view.view3d.from = view.view3d.at + eavlVector3(0,0, camera_direction*ds_size*2);
    view.view3d.up = eavlVector3(0,1,0);
    view.view3d.fov = 0.5;
    view.view3d.size = ds_size; 
    view.view3d.nearplane = ds_size/16.;
    view.view3d.farplane = ds_size*4;
    view.SetupMatrices();
    eavlPoint3 mins(view.minextents[0],view.minextents[1],view.minextents[2]);
    eavlPoint3 maxs(view.maxextents[0],view.maxextents[1],view.maxextents[2]);
    mins = view.V * mins;
    maxs = view.V * maxs;
    float32 far = std::max(-mins.z, -maxs.z);
    float32 near = std::min(-mins.z, -maxs.z); view.view3d.nearplane = near * 0.9f; 
    view.view3d.farplane = far * 1.1f; 
    view.SetupMatrices(); 
    
}

#ifdef PARALLEL
//-----------------------------------------------------------------------------
int *
EAVLPipeline::Renderer::FindVisibilityOrdering(eavlPlot *plot)
{
    //
    // In order for parallel volume rendering to composite correctly,
    // we nee to establish a visibility ordering to pass to IceT.
    // We will transform the data extents into camera space and
    // take the minimum z value. Then sort them while keeping 
    // track of rank, then pass the list in.
    //

    eavlPoint3 min_extents;
    eavlPoint3 max_extents;

    min_extents.x = plot->GetMinCoordExtentOrig(0);
    min_extents.y = plot->GetMinCoordExtentOrig(1);
    min_extents.z = plot->GetMinCoordExtentOrig(2);
    max_extents.x = plot->GetMaxCoordExtentOrig(0);
    max_extents.y = plot->GetMaxCoordExtentOrig(1);
    max_extents.z = plot->GetMaxCoordExtentOrig(2);

    eavlView view = m_window->view;
    //
    // z's should both be negative since the camera is 
    // looking down the neg z-axis
    //
    double x[2], y[2], z[2];
    x[0] = plot->GetMinCoordExtentOrig(0);;
    y[0] = plot->GetMinCoordExtentOrig(1);
    z[0] = plot->GetMinCoordExtentOrig(2);
    x[1] = plot->GetMaxCoordExtentOrig(0);
    y[1] = plot->GetMaxCoordExtentOrig(1);
    z[1] = plot->GetMaxCoordExtentOrig(2);
    
    float minz;
    minz = std::numeric_limits<float>::max();
    eavlPoint3 extent_point;
    
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            for(int k = 0; k < 2; k++)
            {
                extent_point.x = x[i];
                extent_point.y = y[j];
                extent_point.z = z[k];
                extent_point = view.V * extent_point;
                minz = std::min(minz, -extent_point.z);
            }
        }
    }

    int data_type_size;


    MPI_Type_size(MPI_FLOAT, &data_type_size);
    void *z_array;
    
    void *vis_rank_order = malloc(m_mpi_size * sizeof(int));
    Visibility *vis_order;

    if(m_rank == 0)
    {
        // TODO CDH :: new / delete, or use conduit?
        z_array = malloc(m_mpi_size * data_type_size);
    }

    MPI_Gather(&minz, 1, MPI_FLOAT, z_array, 1, MPI_FLOAT, 0, m_mpi_comm);

    if(m_rank == 0)
    {
        vis_order = new Visibility[m_mpi_size];
        
        for(int i = 0; i < m_mpi_size; i++)
        {
            vis_order[i].m_rank = i;
            vis_order[i].m_minz = ((float*)z_array)[i];
        }

        std::qsort(vis_order,
                   m_mpi_size,
                   sizeof(Visibility),
                   CompareVisibility);

        
        for(int i = 0; i < m_mpi_size; i++)
        {
            ((int*) vis_rank_order)[i] = vis_order[i].m_rank;;
        }
        
        free(z_array);
        delete[] vis_order;
    }

    MPI_Bcast(vis_rank_order, m_mpi_size, MPI_INT, 0, m_mpi_comm);
    return (int*)vis_rank_order;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Renderer::SetParallelPlotExtents(eavlPlot * plot)
{
    // We need to get the correct data extents for all processes
    // in order to get the correct color map values
    float64 local_min = plot->GetMinDataExtent();
    float64 local_max = plot->GetMaxDataExtent();
    
    float64 global_min = 0;
    float64 global_max = 0;

    MPI_Allreduce((void *)(&local_min),
                  (void *)(&global_min), 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&local_max),
                  (void *)(&global_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    plot->SetDataExtents(global_min, global_max);

    eavlView &view = m_window->view;
    float64 tmp;
   
    // Set the global spatial bounds
    MPI_Allreduce(&view.minextents[0], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);
    view.minextents[0] = tmp; 

    MPI_Allreduce(&view.minextents[1], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);
    
    view.minextents[1] = tmp; 

    MPI_Allreduce(&view.minextents[2], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);

    view.minextents[2] = tmp; 

    MPI_Allreduce(&view.maxextents[0], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    view.maxextents[0] = tmp; 

    MPI_Allreduce(&view.maxextents[1], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    view.maxextents[1] = tmp; 
    
    MPI_Allreduce(&view.maxextents[2], 
                  &tmp, 
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    view.maxextents[2] = tmp; 
}
#endif

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
EAVLPipeline::EAVLPipeline()
{
    STRAWMAN_BLOCK_TIMER(PIPELINE_CREATE);
}

//-----------------------------------------------------------------------------
EAVLPipeline::~EAVLPipeline()
{
    Cleanup();
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main pipeline interface methods, which are used by the strawman interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
EAVLPipeline::Initialize(const conduit::Node &options)
{

    std::string backend_type = "cpu";


    if(options.has_path("pipeline/backend"))
    {
        backend_type = options["pipeline/backend"].as_string();
    }
    
    if(backend_type == "cpu")
    {
        eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);
    }
    else if(backend_type == "cuda")
    {
        // An alternative would be to set PreferGPU, but it may not execute
        // on the GPU. An error would be generated.
        eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
    }
#if PARALLEL
    if(!options.has_path("mpi_comm"))
    {
        STRAWMAN_ERROR("Missing Strawman::Open options missing MPI communicator (mpi_comm)");
    }

    int mpi_handle = options["mpi_comm"].value();
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_handle);

#ifdef HAVE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0 && device_count <= 256)
    {
        int rank;  
        MPI_Comm_rank(mpi_comm ,&rank);
        int rank_device = rank % device_count;
        //err = cudaSetDevice(rank_device);
        eavlInitializeGPU(rank_device);
        if(false)
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
            STRAWMAN_INFO("Rank "<<rank<<" on "<<proc_name<<" using device "<<rank_device<<"\n");

        }
    }
    else
    {
        STRAWMAN_ERROR("EAVL GPUs is enabled but none found");
    }
#endif


    m_renderer = new EAVLPipeline::Renderer(mpi_comm);
#else
    m_renderer = new EAVLPipeline::Renderer();
#endif
    
    m_renderer->SetOptions(options);
}


//-----------------------------------------------------------------------------
void
EAVLPipeline::Cleanup()
{

    for(int i  = 0; i < m_plots.size(); i++)
    {
        delete m_plots.at(i).m_eavl_dataset;
    }
    m_plots.clear();
    m_renderer->ClearScene();

    // Uncommenting out this line writes all the timers to a file.
    // BlockTimer::WriteLogFile();
    
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Publish(const conduit::Node &data)
{
    m_data.set_external(data);

    m_renderer->SetData(&m_data);
    
    //
    // We need to clear the scene and 
    // the current plot list when the 
    // is re-published;
    //
    
    for(int i  = 0; i < m_plots.size(); i++)
    {
        delete m_plots.at(i).m_eavl_dataset;
    }
    m_plots.clear();
    m_renderer->ClearScene();
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::Execute(const conduit::Node &actions)
{
    //
    // Loop over the actions
    //
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        STRAWMAN_INFO("Executing " << action["action"].as_string());
        
        if (action["action"].as_string() == "add_plot")
        {
            AddPlot(action);
        }
        else if (action["action"].as_string() == "add_filter")
        {
            AddFilter(action);
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
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Helpers for adding Filters to the EAVL Pipeline
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void 
EAVLPipeline::AddFilter(const conduit::Node &node)
{
    if(m_plots.size() < 1)
    {
        STRAWMAN_ERROR("There must be a least one plot to add a filter.");
    }

    if(node["type"].as_string() == "box_filter")
    {
        BoxFilter(node);
    }
    else if(node["type"].as_string() == "isosurface_filter")
    {
        IsosurfaceFilter(node);
    }
    else if(node["type"].as_string() == "cell_to_node_filter")
    {
        CellToNodeFilter(node);
    }
    else if(node["type"].as_string() == "threshold_filter")
    {
        ThresholdFilter(node);
    }
    else if(node["type"].as_string() == "external_faces_filter")
    {
        ExternalFacesFilter();
    }
    else 
    {
        STRAWMAN_INFO( "Warning: Unknown filter type "
                       << node["type"].as_string()
                       <<" Filter not applied.");
        node.print();
    }
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::ExternalFacesFilter()
{
    STRAWMAN_BLOCK_TIMER(EXTERNAL_FACES);
    //
    // Box plot is an EAVL mutator, which adds a cell set to 
    // the plot data set. For now, we assume only a single plot
    //
    Plot *current_plot = &m_plots[0];
    // Get dimensionality
    const int topo_dims = current_plot->m_eavl_dataset->GetCoordinateSystem(0)->GetDimension();
    eavlExternalFaceMutator facer;
    if(topo_dims != 3)
    {
        STRAWMAN_ERROR("EAVL Pipeline error: external Faces filter cell set must be 3d ");
    }
    try
    {
        facer.SetDataSet(current_plot->m_eavl_dataset);
        facer.SetCellSet(current_plot->m_cell_set_name);
        facer.Execute();
    }
    catch (const eavlException &e)
    {
        STRAWMAN_ERROR("EAVL exception while applying external Faces filter "
                        << e.GetErrorText());
    }
    //
    // Create new plot to replace the current one 
    //
    
    //Use eavl's default naming convention
    
    string new_cell_set_name = "extface_of_";
    new_cell_set_name += current_plot->m_cell_set_name;
    
    eavlPlot *newPlot = new eavlPlot(current_plot->m_eavl_dataset,
                                     new_cell_set_name);
    newPlot->SetColorTableByName("Spectral");
    
    string new_var_name = current_plot->m_var_name; 
    
    newPlot->SetField(new_var_name);
    
    delete current_plot->m_eavl_plot;
    current_plot->m_eavl_plot = newPlot;

    //
    // Set the Plot names so another filter can operate
    // on the data if needed.
    //
    current_plot->m_cell_set_name = new_cell_set_name;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::BoxFilter(const conduit::Node &node)
{
    
    //
    // Box plot is a mutator, which adds a cell set to 
    // the plot data set. For now, we assume only a single plot
    //
    Plot *current_plot = &m_plots[0];
    // Get dimensionality
    const int dims = current_plot->m_eavl_dataset->GetCoordinateSystem(0)->GetDimension();
    eavlBoxMutator boxer;
    const float64 *range = node["range"].as_float64_ptr();
    
    try
    {
        boxer.SetDataSet(current_plot->m_eavl_dataset);
        
        if(dims == 2)
        {
            boxer.SetRange2D(range[0], range[1], range[2],range[3]);
        }
        if(dims == 3)
        {
            boxer.SetRange3D(range[0],
                             range[1],
                             range[2],
                             range[3],
                             range[4],
                             range[5]);
        }
        boxer.SetCellSet(current_plot->m_cell_set_name);
        boxer.Execute();
    }
    catch (const eavlException &e)
    {
        STRAWMAN_ERROR("EAVL exception while applying external Faces filter "
                       << e.GetErrorText());
    }
    
    //
    // Create new plot to replace the current one 
    //
    
    //eavl's default naming convention
    
    string new_cell_set_name = "box_of_";
    new_cell_set_name += current_plot->m_cell_set_name;
    
    eavlPlot *new_plot = new eavlPlot(current_plot->m_eavl_dataset,
                                      new_cell_set_name);
    
    new_plot->SetColorTableByName("Spectral");
    
    string new_var_name;
    eavlField *field = current_plot->m_eavl_dataset->GetField(current_plot->m_var_name);
    
    if(field->GetAssociation() == eavlField::ASSOC_CELL_SET)
    {
        new_var_name = "subset_of_";
        new_var_name += current_plot->m_var_name; 
    }
    else 
    {
        new_var_name = current_plot->m_var_name;
    }
    new_plot->SetField(new_var_name);
    
    delete current_plot->m_eavl_plot;
    current_plot->m_eavl_plot = new_plot;

    //
    // Set the Plot names so another filter can operate
    // on the data if needed.
    //
    current_plot->m_var_name = new_var_name;
    current_plot->m_cell_set_name = new_cell_set_name;
}

//-----------------------------------------------------------------------------
void
EAVLPipeline::ThresholdFilter(const conduit::Node &node)
{
    
    //
    // Threshold plot is a mutator, which adds a cell set to 
    // the plot data set. For now, we assume only a single plot
    //
    Plot *current_plot = &m_plots[0];
    // Get dimensionality
    const int dims = current_plot->m_eavl_dataset->GetCoordinateSystem(0)->GetDimension();
    eavlThresholdMutator thresh;
    float64 min_val = node["min_value"].to_float64();
    float64 max_val = node["max_value"].to_float64();
    
    try
    {
        thresh.SetDataSet(current_plot->m_eavl_dataset);
        thresh.SetCellSet(current_plot->m_cell_set_name);
        thresh.SetField(current_plot->m_var_name);
        thresh.SetRange(min_val, max_val);
        thresh.Execute();
    }
    catch (const eavlException &e)
    {
        STRAWMAN_ERROR("EAVL exception while applying a threshold filter "
                       << e.GetErrorText());
    }
    
    //
    // Create new plot to replace the current one 
    //
    
    //eavl's default naming convention
    
    string new_cell_set_name = "threshold_of_";
    new_cell_set_name += current_plot->m_cell_set_name;
    
    eavlPlot *new_plot = new eavlPlot(current_plot->m_eavl_dataset,
                                      new_cell_set_name);
    new_plot->SetColorTableByName("Spectral");
    
    string new_var_name;
    eavlField *field = current_plot->m_eavl_dataset->GetField(current_plot->m_var_name);
    if(field->GetAssociation() == eavlField::ASSOC_CELL_SET)
    {
        new_var_name = "subset_of_";
        new_var_name += current_plot->m_var_name; 
    }
    else 
    {
        new_var_name = current_plot->m_var_name;
    }
    new_plot->SetField(new_var_name);
    
    delete current_plot->m_eavl_plot;
    current_plot->m_eavl_plot = new_plot;
    //
    // Set the Plot names so another filter can operate
    // on the data if needed.
    //
    current_plot->m_var_name = new_var_name;
    current_plot->m_cell_set_name = new_cell_set_name;

}


//-----------------------------------------------------------------------------
void
EAVLPipeline::IsosurfaceFilter(const conduit::Node &node)
{
    //
    // Isosurface will replace the current
    // data set.
    // 
    Plot *current_plot = &m_plots[0];
    eavlIsosurfaceFilter iso;

    try
    {
        iso.SetInput(current_plot->m_eavl_dataset);
        float64 isoValue = node["iso_value"].to_float64();
        iso.SetCellSet(current_plot->m_cell_set_name);
        iso.SetField(current_plot->m_var_name);
        iso.SetIsoValue(isoValue);
        iso.Execute();
        
        if(DEBUG)
        {
            ostringstream oss;
            iso.GetOutput()->PrintSummary(oss);
            STRAWMAN_INFO("Adding Isosurface Filter\n" << oss.str());
        }
    }
    catch (const eavlException &e)
    {
        STRAWMAN_ERROR("EAVL exception while applying a isosurface filter "
                       << e.GetErrorText());
    }
    
    //
    // Create new plot to replace the current one 
    // and follow eavl's default names
    //

    string new_cell_set_name = "iso";
    eavlDataSet *output = iso.GetOutput();
    eavlPlot *new_plot = new eavlPlot(output,
                                      new_cell_set_name);
    delete current_plot->m_eavl_dataset;
    current_plot->m_eavl_dataset = output;
    
    new_plot->SetField(current_plot->m_var_name);
    new_plot->SetColorTableByName("blue");
    
    delete current_plot->m_eavl_plot;
    
    current_plot->m_eavl_plot = new_plot;

    //set the Plot names so another filter can operate
    //on the data if needed.
    current_plot->m_cell_set_name = new_cell_set_name;

}

//-----------------------------------------------------------------------------
void
EAVLPipeline::CellToNodeFilter(const conduit::Node &node)
{
    // For now, we assume only a single plot
    Plot *current_plot = &m_plots[0];
    
    eavlCellToNodeRecenterMutator cell_to_node;
    try
    {
        cell_to_node.SetDataSet(current_plot->m_eavl_dataset);
        cell_to_node.SetField(current_plot->m_var_name);
        cell_to_node.SetCellSet(current_plot->m_cell_set_name);
        cell_to_node.Execute();

        if(DEBUG)
        {
            ostringstream oss;
            current_plot->m_eavl_dataset->PrintSummary(oss);
            STRAWMAN_INFO("Adding CellToNode Filter\n" << oss.str());
        }

    }
    catch (const eavlException &e)
    {
        STRAWMAN_ERROR("EAVL exception while applying a cell-to-node filter "
                       << e.GetErrorText());
    }

    //
    // Create new plot to replace the current one 
    //
    
    eavlPlot *new_plot = new eavlPlot(current_plot->m_eavl_dataset,
                                      current_plot->m_cell_set_name);

    new_plot->SetColorTableByName("Spectral");
    
    //eavls default naming convention
    string new_var_name;
    new_var_name = "nodecentered_";
    new_var_name += current_plot->m_var_name; 
    new_plot->SetField(new_var_name);
    
    delete current_plot->m_eavl_plot;
    current_plot->m_eavl_plot = new_plot;

    //
    // Set the Plot names so another filter can operate
    // on the data if needed.
    //
    current_plot->m_var_name = new_var_name;

}



//-----------------------------------------------------------------------------
void
EAVLPipeline::AddPlot(const conduit::Node &action)
{
    const std::string field_name = action["field_name"].as_string();

    //
    // Create the plot.
    //
    Plot plot;
    plot.m_var_name = field_name;
    plot.m_drawn = false;
    plot.m_hidden = false;
    plot.m_eavl_dataset = DataAdapter::BlueprintToEAVLDataSet(m_data,
                                                              field_name);
    plot.m_eavl_plot    = new eavlPlot(plot.m_eavl_dataset,
                                       plot.m_eavl_dataset->GetCellSet(0)->GetName());

    // we need the topo name ...
    //const Node &n_field  = m_data["fields"][field_name];
    //string topo_name     = n_field["topology"].as_string();
    string topo_name = "cells";
    
    
    plot.m_cell_set_name = topo_name;

    //This will force the plot to get the data extents
    plot.m_eavl_plot->SetField(field_name.c_str());
    plot.m_eavl_plot->SetColorTableByName("Spectral");
   
    if(action.has_path("render_options"))
    {
      plot.m_render_options = action.fetch("render_options"); 
    }
    else 
    {
        plot.m_render_options = conduit::Node(); 
    }
    
    if(DEBUG) 
    {
        ostringstream oss;
        plot.m_eavl_dataset->PrintSummary(oss);
        
        STRAWMAN_INFO("Adding plot " 
                      << plot.m_eavl_dataset->GetCellSet(0)->GetName()
                      << " and variable "<< field_name
                      << " " << oss.str());
    }

    m_plots.push_back(plot);

}

//-----------------------------------------------------------------------------
void
EAVLPipeline::DrawPlots()
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
EAVLPipeline::RenderPlot(const int plot_id, conduit::Node &render_options)
{ 
    
    STRAWMAN_BLOCK_TIMER(SAVE_WINDOW);

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

    Renderer::RenderMode m_render_mode;

    //
    // Determine the render mode, default is opengl / osmesa.
    //

    // 
    //   Note: There is a disconnect here between spacial dimensions and 
    //         topological dimension. Cellset dims are topological.
    //         topo dims is how many dims a cell has (e.g., 2 for a triangle and 3 for a tet
    //         render dims is how many axes the coordinates have (e.g., 2 = 2d plot)
    int topo_dims = m_plots[0].m_eavl_dataset->GetCellSet(m_plots[0].m_cell_set_name)->GetDimensionality();
    const int render_dims = m_plots[0].m_eavl_dataset->GetCoordinateSystem(0)->GetDimension();

    if(render_options.has_path("renderer"))
    {
        if(render_options["renderer"].as_string() == "opengl")
        {
            m_render_mode = Renderer::OPENGL;
        }
        else if(render_options["renderer"].as_string() == "volume")
        {
            m_render_mode = Renderer::VOLUME;
        }
        else if(render_options["renderer"].as_string() == "raytracer")
        {
            m_render_mode = Renderer::RAYTRACER;
        }
        else
        {
            STRAWMAN_INFO( "EAVL Pipeline: unknown renderer "
                           << render_options["Renderer"].as_string() << endl
                           << "Defaulting to OpenGl");
            m_render_mode = Renderer::OPENGL;
        }
    }
    else
    {
        m_render_mode = Renderer::OPENGL;
    }
    // Check to see if the renderer supports the plot type
    if((m_render_mode == Renderer::VOLUME) && (topo_dims != 3))
    {
        STRAWMAN_ERROR("Volume rendering is only supported for 3D data sets.");
    }

    char *image_file_name = NULL;
    
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
    //  For surfaces, apply external faces 
    //  to sigficanly reduce rendering time.
    //  Note: external faces filter in eavl is serial.
    //  
    
    // only apply this when cells are 3d and we
    // want to render the surface
    if((m_render_mode == Renderer::OPENGL ||
       m_render_mode == Renderer::RAYTRACER) && topo_dims == 3)
    {
        ExternalFacesFilter();
    }

    //
    //    Check for camera attributes
    //
    if(render_options.has_path("camera")) 
    {
        m_renderer->SetCameraParams(render_options.fetch("camera"));
    }
        
    //
    // Check for Color Map
    //
    
    if(render_options.has_path("color_map"))
    {
        m_renderer->SetTransferFunctionParams(render_options.fetch("color_map"));
    }
  

    m_renderer->Render(m_plots[plot_id].m_eavl_plot,
                       image_width,
                       image_height,
                       m_render_mode,
                       render_dims,
                       image_file_name);
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end strawman:: --
//-----------------------------------------------------------------------------



