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
/// file: strawman_vtkm_rendering.hpp
///
//-----------------------------------------------------------------------------
#ifndef STRAWMAN_VTKM_RENDERING_HPP
#define STRAWMAN_VTKM_RENDERING_HPP

#include <vtkm/rendering/ColorTable.h>
#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>
#include <vtkm/rendering/MapperVolume.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <conduit.hpp>

#include <strawman_png_encoder.hpp>
#include <strawman_web_interface.hpp>
#include <strawman_logging.hpp>


// mpi related includes
#ifdef PARALLEL
#include <mpi.h>
//----iceT includes 
#include <strawman_icet_compositor.hpp>
//---- conduit mpi 
#include <conduit_relay_mpi.hpp>
#endif


using namespace strawman;
//-----------------------------------------------------------------------------
// -- begin strawman:: --
//-----------------------------------------------------------------------------
namespace strawman
{
//-----------------------------------------------------------------------------
// -- VTKm typedefs for convienince and sanity
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
//
//
//-----------------------------------------------------------------------------
struct VTKMVisibility
{
    int   m_rank;
    float m_minz;
};

enum RendererType 
{
    VOLUME,
    RAYTRACER 
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Internal Class that Handles Rendering via VTKM
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class Renderer
{
public:
      typedef vtkm::rendering::Actor                           vtkmActor;
      typedef vtkm::rendering::Camera                          vtkmCamera;
      typedef vtkm::rendering::Color                           vtkmColor;
      typedef vtkm::rendering::ColorTable                      vtkmColorTable; 
      typedef vtkm::rendering::Canvas                          vtkmCanvas;
      typedef vtkm::rendering::CanvasRayTracer                 vtkmCanvasRayTracer;
      typedef vtkm::rendering::Mapper                          vtkmMapper;
      typedef vtkm::rendering::MapperVolume                    vtkmVolumeRenderer;
      typedef vtkm::rendering::MapperRayTracer                 vtkmRayTracer;
      typedef vtkm::Vec<vtkm::Float32,3>                       vtkmVec3f;
      Renderer();

#ifdef PARALLEL
      Renderer(MPI_Comm mpi_comm);
#endif

      ~Renderer();
  
      void SetOptions(const conduit::Node &options);

      void SetTransferFunction(const conduit::Node &tFunction);
      void CreateDefaultTransferFunction(vtkmColorTable &color_table);
      void SetCamera(const conduit::Node &_camera);
      void SetData(conduit::Node *data_ptr);
  
      void ClearScene();

      void Render(vtkmActor *&plot,
                  int image_height,
                  int image_width, 
                  RendererType type,
                  int dims,
                  const char *image_file_name = NULL);
 
      // TODO: Move to pipeline?
      void WebSocketPush(PNGEncoder &png);
      void WebSocketPush(const std::string &img_file_path);
      void SaveImage(const char *image_file_name);  
private:

//-----------------------------------------------------------------------------
// private structs // classes
//-----------------------------------------------------------------------------

  struct RenderParams
  {
    public:
       int          m_height;
       int          m_width;
       RendererType m_render_type;
       int          m_plot_dims;

       RenderParams()
       : m_height(-1),
         m_width(-1),
         m_render_type(RAYTRACER),
         m_plot_dims(-1)
       {}
  };

//-----------------------------------------------------------------------------
// private methods
//-----------------------------------------------------------------------------
    void Init();
    void Cleanup();
    void NullRendering();
    void ResetViewPlanes();
    void InitRendering(int plotDims);
    void SetTransferFunction(conduit::Node &tfunction, 
                             vtkmColorTable *tf);
    void SetCameraAttributes(conduit::Node &node);
    void SetDefaultCameraView(vtkmActor *plot);
    void SetupCamera();
    vtkmColorTable  SetColorMapFromNode();
//-----------------------------------------------------------------------------
// private methods for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL
    void  CheckIceTError();
    int  *FindVisibilityOrdering(vtkmActor *plot);
    void  SetParallelPlotExtents(vtkmActor * plot);
#endif
  

//-----------------------------------------------------------------------------
// private data members
//-----------------------------------------------------------------------------

    vtkmCanvas         *m_canvas;
    vtkmMapper         *m_renderer;
    vtkmCamera         *m_vtkm_camera;

    vtkmColor           m_bg_color;
    vtkm::Bounds        m_spatial_bounds; 
    RendererType        m_render_type;
    RenderParams        m_last_render;
  
    conduit::Node       m_transfer_function;
    conduit::Node       m_camera;
  
    conduit::Node      *m_data;

    // always keep rank, even for serial
    int                 m_rank;
  
    conduit::Node       m_options;              // CDH: need to store?
    bool                m_web_stream_enabled;   // CDH: move to pipeline ?
    WebInterface        m_web_interface;        // CDH: move to pipeline ?
  
    PNGEncoder          m_png_data;

//-----------------------------------------------------------------------------
// private vars for MPI case
//-----------------------------------------------------------------------------
#ifdef PARALLEL
    MPI_Comm            m_mpi_comm;
    
    IceTCompositor      m_icet;
    
    int                 m_mpi_size;

static
int
VTKMCompareVisibility(const void *a, const void *b)
{
  if((*(VTKMVisibility*)a).m_minz <  (*(VTKMVisibility*)b).m_minz)  return -1;
  if((*(VTKMVisibility*)a).m_minz == (*(VTKMVisibility*)b).m_minz)  return 0;
  if((*(VTKMVisibility*)a).m_minz >  (*(VTKMVisibility*)b).m_minz)  return 1;
  // TODO: default return?
  return -1;
}
#endif 


};

}; //namespace strawman
//-----------------------------------------------------------------------------
// -- end strawman:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
