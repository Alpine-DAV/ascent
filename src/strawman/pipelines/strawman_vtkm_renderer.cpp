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
/// file: strawman_vtkm_renderer.cpp
///
//-----------------------------------------------------------------------------

#warning "Should not have to define this"
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL
// this should be defined in the file below
//#include <vtkm/cont/internal/DeviceAdapterTag.h>

#include "strawman_vtkm_renderer.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>

// other strawman includes
#include <strawman_block_timer.hpp>
#include <strawman_png_encoder.hpp>
#include <strawman_web_interface.hpp>

using namespace std;
using namespace conduit;
namespace strawman {
template<typename T>
T *
GetVTKMPointer(vtkm::cont::ArrayHandle<T> &handle)
{
  typedef typename vtkm::cont::ArrayHandle<T> HandleType;
  typedef typename HandleType::template ExecutionTypes<vtkm::cont::DeviceAdapterTagSerial>::Portal PortalType;
  typedef typename vtkm::cont::ArrayPortalToIterators<PortalType>::IteratorType IteratorType;
  IteratorType iter = vtkm::cont::ArrayPortalToIterators<PortalType>(handle.GetPortalControl()).GetBegin();
  return &(*iter);
}
//-----------------------------------------------------------------------------
// Renderer public methods
//-----------------------------------------------------------------------------

Renderer::Renderer()
{   
    Init();
    NullRendering();
    m_rank  = 0; 
}


//-----------------------------------------------------------------------------
#ifdef PARALLEL
//-----------------------------------------------------------------------------

Renderer::Renderer(MPI_Comm mpi_comm)
: m_mpi_comm(mpi_comm)
{
    Init();
    NullRendering();
    m_icet.Init(m_mpi_comm);

    MPI_Comm_rank(m_mpi_comm, &m_rank);
    MPI_Comm_size(m_mpi_comm, &m_mpi_size);
}

//-----------------------------------------------------------------------------
// Renderer private methods
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------

void
Renderer::Init()
{
    m_camera.reset();
    m_transfer_function.reset();

    m_bg_color.Components[0] = 1.0f;
    m_bg_color.Components[1] = 1.0f;
    m_bg_color.Components[2] = 1.0f;
    m_bg_color.Components[3] = 1.0f;

    m_web_stream_enabled = false;
}

//-----------------------------------------------------------------------------

void
Renderer::NullRendering()
{
    m_canvas       = NULL;
    m_renderer     = NULL;
    m_vtkm_camera  = NULL;
}

//-----------------------------------------------------------------------------

void
Renderer::Cleanup()
{
    
    if(m_canvas)
    {
        delete m_canvas;
    }

    if(m_renderer)
    {
        delete m_renderer;
    }

    if(m_vtkm_camera)
    {
        delete m_vtkm_camera;
    }
    
    NullRendering();
}

//-----------------------------------------------------------------------------

void
Renderer::InitRendering(int plot_dims)
{
    if(plot_dims != 2 && plot_dims != 3)
    {
        STRAWMAN_ERROR("VTKM rendering currently only supports 3D");
    }
    STRAWMAN_BLOCK_TIMER(RENDER_INIT);
    
    // start from scratch
    Cleanup();
    

    //Insert code for vtkmScene and annotators here

    //
    // Create the appropriate renderer
    //      
    m_renderer = NULL;

    if(m_render_type == VOLUME)
    {
        m_renderer = new vtkmVolumeRenderer();
    }
    else if(m_render_type == RAYTRACER)
    {   
        m_renderer = new vtkmRayTracer();
    }
  
    if(m_renderer == NULL)
    {
        STRAWMAN_ERROR("vtkmMapper was not created");
    }

    m_canvas = new vtkmCanvasRayTracer(1024,1024);
    m_canvas->SetBackgroundColor(m_bg_color);

    if(m_canvas == NULL)
    {
      STRAWMAN_ERROR("vtkmCanvas was not created.");
    }
    
    m_vtkm_camera = new vtkmCamera;
    
    if(m_canvas == NULL)
    {
      STRAWMAN_ERROR("vtkmCamera was not created.");
    }
}

//-----------------------------------------------------------------------------

void
Renderer::SetDefaultCameraView(vtkmActor *plot)
{
    STRAWMAN_BLOCK_TIMER(SET_CAMERA)

   
    // Set some defaults
    m_canvas->ResizeBuffers(1024, 1024);
    m_spatial_bounds = plot->GetSpatialBounds(); 
#ifdef PARALLEL
    // Rank plot extents set when plot is created.
    // We need to perfrom global reductions to create
    // the same view on every rank.
    vtkm::Float64 x_min = plot->GetSpatialBounds().X.Min;
    vtkm::Float64 x_max = plot->GetSpatialBounds().X.Max;
    vtkm::Float64 y_min = plot->GetSpatialBounds().Y.Min;
    vtkm::Float64 y_max = plot->GetSpatialBounds().Y.Max;
    vtkm::Float64 z_min = plot->GetSpatialBounds().Z.Min;
    vtkm::Float64 z_max = plot->GetSpatialBounds().Z.Max;
    vtkm::Float64 global_x_min = 0;
    vtkm::Float64 global_x_max = 0;
    vtkm::Float64 global_y_min = 0;
    vtkm::Float64 global_y_max = 0;
    vtkm::Float64 global_z_min = 0;
    vtkm::Float64 global_z_max = 0;

    MPI_Allreduce((void *)(&x_min),
                  (void *)(&global_x_min), 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&x_max),
                  (void *)(&global_x_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&y_min),
                  (void *)(&global_y_min), 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&y_max),
                  (void *)(&global_y_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&z_min),
                  (void *)(&global_z_min), 
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  m_mpi_comm);

    MPI_Allreduce((void *)(&z_max),
                  (void *)(&global_z_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  m_mpi_comm);

    m_spatial_bounds.X.Min = global_x_min;
    m_spatial_bounds.X.Max = global_x_max;
    m_spatial_bounds.Y.Min = global_y_min;
    m_spatial_bounds.Y.Max = global_y_max;
    m_spatial_bounds.Z.Min = global_z_min;
    m_spatial_bounds.Z.Max = global_z_max;
#endif
    vtkmVec3f total_extent;
    total_extent[0] = vtkm::Float32(m_spatial_bounds.X.Max - m_spatial_bounds.X.Min);
    total_extent[1] = vtkm::Float32(m_spatial_bounds.Y.Max - m_spatial_bounds.Y.Min);
    total_extent[2] = vtkm::Float32(m_spatial_bounds.Z.Max - m_spatial_bounds.Z.Min);
    vtkm::Float32 mag = vtkm::Magnitude(total_extent);
    vtkmVec3f n_total_extent = total_extent;
    vtkm::Normalize(n_total_extent);
    
    vtkmVec3f bounds_min(m_spatial_bounds.X.Min,
                         m_spatial_bounds.Y.Min,
                         m_spatial_bounds.Z.Min);
    
    //
    // detect a 2d data set
    //
    int min_dim = 0;
    if(total_extent[1] < total_extent[min_dim]) min_dim = 1;
    if(total_extent[2] < total_extent[min_dim]) min_dim = 2;
  
    bool is_2d = (total_extent[min_dim] == 0.f);
    // look at the center
    m_vtkm_camera->SetLookAt(bounds_min + n_total_extent * (mag * 0.5f));
    // find the maximum dim that will be the x in image space
    int x_dim = 0;
    if(total_extent[1] > total_extent[x_dim]) x_dim = 1;
    if(total_extent[2] > total_extent[x_dim]) x_dim = 2;
    
    // choose up to be the other dimension
    vtkmVec3f up(0,0,0);
    int up_dim = 0;
    for(int i = 0; i < 3; ++i)
    {
        if(i != x_dim && i != min_dim) up_dim = i;
    }
    up[up_dim] = 1.f;

    const float default_fov = m_vtkm_camera->GetFieldOfView(); 
    
    vtkmVec3f position(0,0,0);
    if(is_2d)
    {
        vtkmVec3f pos(0.f, 0.f, 0.f);
        for(int i = 0; i < 3; ++i) 
            pos[i] = (total_extent[i] != 0.f) ? bounds_min[i] + total_extent[i] / 2.f : total_extent[i];
        const float pi = 3.14159f;
        float theta = (default_fov + 4) * (pi/180.f);
        float min_pos = std::tan(theta) * total_extent[x_dim] / 2.f;
        m_vtkm_camera->SetLookAt(pos);
        pos[min_dim] = bounds_min[min_dim] + min_pos;
        m_vtkm_camera->SetPosition(pos);
        position = pos;
    }
    else
    {
        position = -n_total_extent * (mag * 1.6);
        position[0] += .001f;
        position[1] += .001f;
        position[2] += .05f*mag;
        m_vtkm_camera->SetPosition(position);
    }
    // set a default near and far plane
    vtkmVec3f bounding_box[8];
    bounding_box[0][0] = m_spatial_bounds.X.Min;
    bounding_box[0][1] = m_spatial_bounds.Y.Min;
    bounding_box[0][2] = m_spatial_bounds.Z.Min;

    bounding_box[1][0] = m_spatial_bounds.X.Min;
    bounding_box[1][1] = m_spatial_bounds.Y.Min;
    bounding_box[1][2] = m_spatial_bounds.Z.Max;

    bounding_box[2][0] = m_spatial_bounds.X.Min;
    bounding_box[2][1] = m_spatial_bounds.Y.Max;
    bounding_box[2][2] = m_spatial_bounds.Z.Min;

    bounding_box[3][0] = m_spatial_bounds.X.Min;
    bounding_box[3][1] = m_spatial_bounds.Y.Max;
    bounding_box[3][2] = m_spatial_bounds.Z.Max;

    bounding_box[4][0] = m_spatial_bounds.X.Max;
    bounding_box[4][1] = m_spatial_bounds.Y.Min;
    bounding_box[4][2] = m_spatial_bounds.Z.Min;

    bounding_box[5][0] = m_spatial_bounds.X.Max;
    bounding_box[5][1] = m_spatial_bounds.Y.Min;
    bounding_box[5][2] = m_spatial_bounds.Z.Max;

    bounding_box[6][0] = m_spatial_bounds.X.Max;
    bounding_box[6][1] = m_spatial_bounds.Y.Max;
    bounding_box[6][2] = m_spatial_bounds.Z.Min;

    bounding_box[7][0] = m_spatial_bounds.X.Max;
    bounding_box[7][1] = m_spatial_bounds.Y.Max;
    bounding_box[7][2] = m_spatial_bounds.Z.Max;

    vtkm::Float32 max_distance = 0.01f; 
    for(int i = 0; i < 8; ++i)
    {
        vtkm::Float32 distance = vtkm::Magnitude(bounding_box[i] - position);
        max_distance = vtkm::Max(max_distance, distance);
    }
    max_distance *= 1.1f;

    vtkm::Range clipping_range;
    clipping_range.Min = .01f;
    clipping_range.Max = max_distance;
    m_vtkm_camera->SetClippingRange(clipping_range);
}

//-----------------------------------------------------------------------------
// imp EAVLPipeline::Renderer private methods for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL

//-----------------------------------------------------------------------------

int *
Renderer::FindVisibilityOrdering(vtkmActor *plot)
{
    //
    // In order for parallel volume rendering to composite correctly,
    // we nee to establish a visibility ordering to pass to IceT.
    // We will transform the data extents into camera space and
    // take the minimum z value. Then sort them while keeping 
    // track of rank, then pass the list in.
    //
    vtkm::Matrix<vtkm::Float32,4,4> view_matrix = 
        m_vtkm_camera->CreateViewMatrix();
    
    //
    // z's should both be negative since the camera is 
    // looking down the neg z-axis
    //
    double x[2], y[2], z[2];

    x[0] = plot->GetSpatialBounds().X.Min;
    x[1] = plot->GetSpatialBounds().X.Max;
    y[0] = plot->GetSpatialBounds().Y.Min;
    y[1] = plot->GetSpatialBounds().Y.Max;
    z[0] = plot->GetSpatialBounds().Z.Min;
    z[1] = plot->GetSpatialBounds().Z.Max;
    
    float minz;
    minz = std::numeric_limits<float>::max();
    vtkm::Vec<vtkm::Float32,4> extent_point;
    
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++)
            for(int k = 0; k < 2; k++)
            {
                extent_point[0] = static_cast<vtkm::Float32>(x[i]);
                extent_point[1] = static_cast<vtkm::Float32>(y[j]);
                extent_point[2] = static_cast<vtkm::Float32>(z[k]);
                extent_point[3] = 1.f;
                extent_point = vtkm::MatrixMultiply(view_matrix, extent_point);
                // perform the perspective divide
                extent_point[2] = extent_point[2] / extent_point[3];
                minz = std::min(minz, -extent_point[2]);
            }

    int data_type_size;


    MPI_Type_size(MPI_FLOAT, &data_type_size);
    void *z_array;
    
    void *vis_rank_order = malloc(m_mpi_size * sizeof(int));
    VTKMVisibility *vis_order;

    if(m_rank == 0)
    {
        // TODO CDH :: new / delete, or use conduit?
        z_array = malloc(m_mpi_size * data_type_size);
    }

    MPI_Gather(&minz, 1, MPI_FLOAT, z_array, 1, MPI_FLOAT, 0, m_mpi_comm);

    if(m_rank == 0)
    {
        vis_order = new VTKMVisibility[m_mpi_size];
        
        for(int i = 0; i < m_mpi_size; i++)
        {
            vis_order[i].m_rank = i;
            vis_order[i].m_minz = ((float*)z_array)[i];
        }

        std::qsort(vis_order,
                   m_mpi_size,
                   sizeof(VTKMVisibility),
                   VTKMCompareVisibility);

        
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
Renderer::SetParallelPlotExtents(vtkmActor * plot)
{
    STRAWMAN_BLOCK_TIMER(PARALLEL_PLOT_EXTENTS)
    // We need to get the correct data extents for all processes
    // in order to get the correct color map values
    float64 local_min = plot->GetScalarRange().Min;
    float64 local_max = plot->GetScalarRange().Max;
    
    float64 global_min = 0;
    float64 global_max = 0;

    // TODO CDH: use conduit MPI?
    
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
    vtkm::Range scalar_range;
    scalar_range.Min = global_min;
    scalar_range.Max = global_max;
    plot->SetScalarRange(scalar_range);
}
#endif

//-----------------------------------------------------------------------------
// Renderer public methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

Renderer::~Renderer()
{
    STRAWMAN_BLOCK_TIMER(RENDERER_ON_DESTROY);
    
    Cleanup();

#ifdef PARALLEL
    m_icet.Cleanup();
#endif
}

//-----------------------------------------------------------------------------

void
Renderer::SetOptions(const Node &options)
{

    if(options.has_path("web/stream") && 
       options["web/stream"].as_string() == "true")
    {
        m_web_stream_enabled = true;
    }
    
    
}
//-----------------------------------------------------------------------------

void
Renderer::CreateDefaultTransferFunction(vtkmColorTable &color_table)
{
    const vtkm::Int32 num_opacity_points = 256;
    const vtkm::Int32 num_peg_points = 8;
    unsigned char char_opacity[num_opacity_points] = { 1,1,1,1,1,1,1,0,
                                                       5,5,5,5,5,5,5,5,
                                                       7,7,7,7,7,7,7,7,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       12,12,12,12,12,12,12,12,
                                                       100,100,100,100,100,100,100,100,
                                                       75,75,75,75,75,75,75,75,
                                                       75,75,75,75,75,75,75,75,
                                                       75,75,75,75,75,75,75,75,
                                                       75,75,75,75,75,75,75,75,
                                                       55,55,55,55,55,55,55,55 };                                         

    char *char_opacity_factor = std::getenv("VTKm_OPACITY_FACTOR");
    float opacity_factor = 0.1f;
    if(char_opacity_factor != 0)
    {
      opacity_factor = atof(char_opacity_factor);
    }
    for (int i = 0; i < num_opacity_points; ++i)
    {
        vtkm::Float32 position = vtkm::Float32(i) / vtkm::Float32(num_opacity_points);
        vtkm::Float32 value = vtkm::Float32(char_opacity[i] / vtkm::Float32(255));
        value *= opacity_factor;
        color_table.AddAlphaControlPoint(position, value);
    }

    unsigned char control_point_colors[num_peg_points*3] = { 
           128, 0, 128, 
           0, 128, 128,
           128, 128, 0, 
           128, 128, 128, 
           255, 255, 0, 
           255, 96, 0, 
           107, 0, 0, 
           224, 76, 76 
       };

    vtkm::Float32 control_point_positions[num_peg_points] = { 0.f, 0.543f, 0.685f, 0.729f,
                                                            0.771f, 0.804f, 0.857f, 1.0f };

    for (int i = 0; i < num_peg_points; ++i)
    {
        vtkm::Float32 position = vtkm::Float32(i) / vtkm::Float32(num_opacity_points);
        vtkm::rendering::Color color;
        color.Components[0] = vtkm::Float32(control_point_colors[i*3+0] / vtkm::Float32(255));
        color.Components[1] = vtkm::Float32(control_point_colors[i*3+1] / vtkm::Float32(255));
        color.Components[2] = vtkm::Float32(control_point_colors[i*3+2] / vtkm::Float32(255));
        color_table.AddControlPoint(control_point_positions[i], color);
    }
}

//-----------------------------------------------------------------------------

void
Renderer::SetData(Node *data_node_ptr)
{
     m_data = data_node_ptr;
}
//-----------------------------------------------------------------------------

void
Renderer::SetTransferFunction(const Node &transfer_function_params)
{
    m_transfer_function.reset();
    m_transfer_function.set(transfer_function_params);
}
//-----------------------------------------------------------------------------

vtkm::rendering::ColorTable
Renderer::SetColorMapFromNode()
{

    std::string color_map_name = "";
    if(m_transfer_function.has_child("name"))
    {
        color_map_name = m_transfer_function["name"].as_string();
    }

    vtkmColorTable color_map(color_map_name);

    if(color_map_name == "")
    {
        color_map.Clear();
    }
    
    if(!m_transfer_function.has_child("control_points"))
    {
        if(color_map_name == "") 
          STRAWMAN_ERROR("Error: a color map node was provided without a color map name or control points");
        return color_map;
    }
    
    NodeConstIterator itr = m_transfer_function.fetch("control_points").children();
    while(itr.has_next())
    {
        const Node &peg = itr.next();
        if(!peg.has_child("position"))
        {
            peg.print();
            STRAWMAN_WARN("Color map control point must have a position. Ignoring");
        }
        float64 position = peg["position"].as_float64();
        
        if(position > 1.0 || position < 0.0)
        {
              STRAWMAN_WARN("Cannot add color map control point position "
                            << position 
                            << ". Must be a normalized scalar.");
        }
  
        if (peg["type"].as_string() == "rgb")
        {
            const float64 *color = peg["color"].as_float64_ptr();
            
            vtkm::rendering::Color ecolor(color[0], color[1], color[2]);
            
            color_map.AddControlPoint(position, ecolor);
        }
        else if (peg["type"].as_string() == "alpha")
        {
            float64 alpha = peg["alpha"].to_float64();
            
            color_map.AddAlphaControlPoint(position, alpha);
        }
        else
        {
            STRAWMAN_WARN("Unknown control point type " << peg["type"].as_string());
        }
    }

    return color_map;
}

//-----------------------------------------------------------------------------

void
Renderer::SetCamera(const Node &camera_params)
{
    m_camera.set(camera_params);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------

void
Renderer::WebSocketPush(PNGEncoder &png)
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
    
    Node status;
    status["type"] = "status";
    status["state"] = 1;
    status["domain"] = 1;
    //status = m_data->fetch("state");
    //status.remove("domain");
    status["data/ndomains"] = ndomains;
    
    m_web_interface.PushMessage(status);
    m_web_interface.PushImage(png);
     

 }

 //-----------------------------------------------------------------------------

void
Renderer::WebSocketPush(const std::string &img_file_path)
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
    

    Node status;
    status["type"] = "status";
    status = m_data->fetch("state");
    status.remove("domain");
    status["data/ndomains"] = ndomains;
    status.print();
    std::string img_file_path_full(img_file_path);
    img_file_path_full = img_file_path_full + ".png";
    m_web_interface.PushMessage(status);
    m_web_interface.PushImage(img_file_path_full);

 }

//-----------------------------------------------------------------------------

void
Renderer::SaveImage(const char *image_file_name)
{
#ifdef PARALLEL
    if(m_rank == 0)
    {
        string ofname(image_file_name);
        ofname +=  ".png";
        m_png_data.Save(ofname);
    }
#else 
    string ofname(image_file_name);
    ofname +=  ".png";
    m_png_data.Save(ofname);
#endif
}

//-----------------------------------------------------------------------------

void
Renderer::Render(vtkmActor *&plot,
                 int image_height,
                 int image_width,
                 RendererType mode,
                 int dims,
                 const char *image_file_name)
{
    STRAWMAN_BLOCK_TIMER(RENDER)
    try
    {
        PNGEncoder png;
        //
        // Do some check to see if we need
        // to re-init rendering
        //

        m_render_type = mode;

               
        bool render_dirty = false;
        bool screen_dirty = false;
        
        if(m_render_type != m_last_render.m_render_type)
        {
            render_dirty = true;
        }
        
        if(dims != m_last_render.m_plot_dims)
        {
            render_dirty = true;
        }
        
        if(image_height != m_last_render.m_height ||
           image_width  != m_last_render.m_width)
        {
            screen_dirty = true;
        }
        
        m_last_render.m_render_type     = m_render_type;
        m_last_render.m_plot_dims       = dims;
        m_last_render.m_height          = image_height;
        m_last_render.m_width           = image_width;

        if(render_dirty)
        {
            InitRendering(dims);
        }
        
        // Set the Default camera position
        SetDefaultCameraView(plot);
        
        if(screen_dirty)
        {
            delete m_canvas;
            m_canvas = new vtkmCanvasRayTracer(image_width,image_height);
            m_canvas->SetBackgroundColor(m_bg_color);
        }
        
        //
        // Check to see if we have camera params
        //
        if(!m_camera.dtype().is_empty())
        {
            SetupCamera();
        } 
       
        //
        // Check for transfer function / color table
        //
        if(!m_transfer_function.dtype().is_empty())
        {
          vtkmColorTable color_table = SetColorMapFromNode();
          vtkmActor *new_actor = new vtkmActor(plot->GetCells(),
                                               plot->GetCoordiantes(),
                                               plot->GetScalarField(),
                                               color_table);
          delete plot;
          plot = new_actor;
        }
        else
        {
            //
            //  Add some opacity if the plot is a volume 
            //  and we have a default color table
            //
            if(m_render_type == VOLUME)
            {
                vtkmColorTable color_table;
                CreateDefaultTransferFunction(color_table);
                vtkmActor *new_actor = new vtkmActor(plot->GetCells(),
                                                     plot->GetCoordiantes(),
                                                     plot->GetScalarField(),
                                                     color_table);
                delete plot;
                plot = new_actor;
            }
        }

        //
        //  We need to set a sample distance for volume plots
        // 
        if(m_render_type == VOLUME)
        {

              //set sample distance
              const vtkm::Float32 num_samples = 200.f;
              vtkm::Vec<vtkm::Float32,3> totalExtent;
              totalExtent[0] = vtkm::Float32(m_spatial_bounds.X.Max - m_spatial_bounds.X.Min);
              totalExtent[1] = vtkm::Float32(m_spatial_bounds.Y.Max - m_spatial_bounds.Y.Min);
              totalExtent[2] = vtkm::Float32(m_spatial_bounds.Z.Max - m_spatial_bounds.Z.Min);
              vtkm::Float32 sample_distance = vtkm::Magnitude(totalExtent) / num_samples;
              vtkmVolumeRenderer *volume_renderer = static_cast<vtkmVolumeRenderer*>(m_renderer);
              
              volume_renderer->SetSampleDistance(sample_distance);
#ifdef PARALLEL
              // Turn of background compositing 
              volume_renderer->SetCompositeBackground(false);
#endif
          }
         
#ifdef PARALLEL
        SetParallelPlotExtents(plot);
        
        //
        //  We need to turn off the background for the
        //  parellel volume render BEFORE the scene
        //  is painted. 
        
        
        int *vis_order = NULL;
        if(m_render_type == VOLUME)
        {
            // Set the backgound color to transparent
            vtkmColor color = m_canvas->GetBackgroundColor();
            
            color.Components[3] = 0.f;
            m_canvas->SetBackgroundColor(color);

            //
            // Calculate visibility ordering AFTER 
            // the camera parameters have been set
            // IceT uses this list to composite the images
            
            //
            // TODO: This relies on plot 0
            //
            vis_order = FindVisibilityOrdering(plot);
    
        }
#endif
        //---------------------------------------------------------------------
        {// open block for RENDER_PAINT Timer
        //---------------------------------------------------------------------
            STRAWMAN_BLOCK_TIMER(RENDER_PAINT);

            m_canvas->Clear();
            plot->Render(*m_renderer, *m_canvas, *m_vtkm_camera);

        //---------------------------------------------------------------------
        } // close block for RENDER_PAINT Timer
        //---------------------------------------------------------------------
        
        //Save the image.
#ifdef PARALLEL

        const float *result_color_buffer = NULL;
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

             
            const float *input_color_buffer  = NULL;
            const float *input_depth_buffer  = NULL;    
            
          
            input_color_buffer = &GetVTKMPointer(m_canvas->GetColorBuffer())[0][0];
            input_depth_buffer = GetVTKMPointer(m_canvas->GetDepthBuffer());
            float bg_color[4];
            bg_color[0] = m_bg_color.Components[0];
            bg_color[1] = m_bg_color.Components[1];
            bg_color[2] = m_bg_color.Components[2];
            bg_color[3] = m_bg_color.Components[3];
            if(m_render_type != VOLUME)
            {   
                result_color_buffer = m_icet.Composite(image_width,
                                                       image_height,
                                                       input_color_buffer,
                                                       input_depth_buffer,
                                                       view_port,
                                                       bg_color);
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
                                                       bg_color);
                // leak?
                free(vis_order);
            }
        
        //---------------------------------------------------------------------
        }// close block for RENDER_COMPOSITE Timer
        //---------------------------------------------------------------------
          
                
        //---------------------------------------------------------------------
        {// open block for RENDER_ENCODE Timer
        //---------------------------------------------------------------------
          
        
        STRAWMAN_BLOCK_TIMER(RENDER_ENCODE);
        //
        // encode the composited image
        //
        if(m_rank == 0)
        {   
            m_png_data.Encode(result_color_buffer,
                              image_width,
                              image_height);
        }
        
        //---------------------------------------------------------------------
        }// close block for RENDER_ENCODE Timer
        //---------------------------------------------------------------------
          

#else
        m_png_data.Encode(&(m_canvas->ColorBuffer[0]),
                          image_width,
                          image_height);
#endif
    

#if PARALLEL
        // png will be null if rank !=0, thats fine
        WebSocketPush(m_png_data);
#else
        WebSocketPush(m_png_data);
#endif

        if(image_file_name != NULL) SaveImage(image_file_name);
    }// end try
    catch (vtkm::cont::Error error) 
    {
      std::cout << "VTK-m Renderer Got the unexpected error: " << error.GetMessage() << std::endl;
    }
}
//-----------------------------------------------------------------------------

void
Renderer::SetupCamera()
{
    //
    // Get the optional camera parameters
    //
    if(m_camera.has_child("look_at"))
    {
        float64 *coords = m_camera["look_at"].as_float64_ptr();
        vtkmVec3f look_at(coords[0], coords[1], coords[2]);
        m_vtkm_camera->SetLookAt(look_at);  
    }
    if(m_camera.has_child("position"))
    {
        float64 *coords = m_camera["position"].as_float64_ptr();
        vtkmVec3f position(coords[0], coords[1], coords[2]);
        m_vtkm_camera->SetPosition(position);  
    }
    
    if(m_camera.has_child("up"))
    {
        float64 *coords = m_camera["up"].as_float64_ptr();
        vtkmVec3f up(coords[0], coords[1], coords[2]);
        vtkm::Normalize(up);
        m_vtkm_camera->SetViewUp(up);
    }
    
    if(m_camera.has_child("fov"))
    {
        m_vtkm_camera->SetFieldOfView(m_camera["fov"].to_float64());
    }

    if(m_camera.has_child("xpan") || m_camera.has_child("ypan"))
    {
        vtkm::Float64 xpan = 0.;
        vtkm::Float64 ypan = 0.;
        if(m_camera.has_child("xpan")) xpan = m_camera["xpan"].to_float64();
        if(m_camera.has_child("ypan")) xpan = m_camera["ypan"].to_float64();
        m_vtkm_camera->Pan(xpan, ypan);
    }

    if(m_camera.has_child("zoom"))
    {
        m_vtkm_camera->Zoom(m_camera["zoom"].to_float64());
    }

    if(m_camera.has_child("nearplane"))
    {
        vtkm::Range clipping_range = m_vtkm_camera->GetClippingRange();
        clipping_range.Min = m_camera["nearplane"].to_float64();
        m_vtkm_camera->SetClippingRange(clipping_range);
    }

    if(m_camera.has_child("farplane"))
    {
        vtkm::Range clipping_range = m_vtkm_camera->GetClippingRange();
        clipping_range.Max = m_camera["farplane"].to_float64();
        m_vtkm_camera->SetClippingRange(clipping_range);
    }
}

}; //namespace strawman

