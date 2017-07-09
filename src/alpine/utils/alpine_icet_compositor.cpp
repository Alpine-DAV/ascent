//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: alpine_icet_compositor.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_icet_compositor.hpp"

#include "alpine_logging.hpp"

 
//-----------------------------------------------------------------------------
#define CHECK_ICET_ERROR( msg )                                                \
{                                                                              \
    if(icetGetError() != ICET_NO_ERROR )                                       \
    {                                                                          \
        ALPINE_WARN("IceT Error Occurred!");                                 \
    }                                                                          \
}                                                                              \



//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
IceTCompositor::IceTCompositor()
: m_rank(0)
{}
  
//-----------------------------------------------------------------------------
IceTCompositor::~IceTCompositor()
{
    // TODO: cleanup?
}

//-----------------------------------------------------------------------------
void
IceTCompositor::Init(MPI_Comm mpi_comm)
{
    m_icet_comm    = icetCreateMPICommunicator(mpi_comm);
    m_icet_context = icetCreateContext(m_icet_comm);
    MPI_Comm_rank(mpi_comm, &m_rank);
}

//-----------------------------------------------------------------------------
unsigned char *
IceTCompositor::Composite(int            width,
                          int            height,
                          const unsigned char *color_buffer,
                          const int           *vis_order,
                          const float         *bg_color)
{
    icetResetTiles();
    icetAddTile(0, 0, width, height, 0);
    
    //best strategy for use with a single tile (i.e., one monitor)
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); 
    icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);

    CHECK_ICET_ERROR();
    
    // is this necessary?
    IceTFloat icet_bg_color[4] = { bg_color[0],
                                   bg_color[1],
                                   bg_color[2],
                                   bg_color[3] };

    icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
    icetEnable(ICET_ORDERED_COMPOSITE);
    icetCompositeOrder(vis_order);
    m_icet_image = icetCompositeImage(color_buffer,
                                      NULL,
                                      NULL,
                                      NULL,
                                      NULL,
                                      icet_bg_color);
    CHECK_ICET_ERROR();
    GetTimings();
    
    unsigned char * res= NULL;
    if(m_rank == 0)
    {
        res = icetImageGetColorub(m_icet_image);
    }
    return res;
}

//-----------------------------------------------------------------------------
unsigned char*
IceTCompositor::Composite(int            width,
                          int            height,
                          const float   *color_buffer,
                          const int     *vis_order,
                          const float   *bg_color)
{
    icetResetTiles();
    icetAddTile(0, 0, width, height, 0);
    unsigned char * ubytes = ConvertBuffer(color_buffer, width * height * 4);     
    //best strategy for use with a single tile (i.e., one monitor)
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); 
    icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);

    CHECK_ICET_ERROR();
    
    // is this necessary?
    IceTFloat icet_bg_color[4] = { bg_color[0],
                                   bg_color[1],
                                   bg_color[2],
                                   bg_color[3] };

    icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
    icetEnable(ICET_ORDERED_COMPOSITE);
    icetCompositeOrder(vis_order);
    m_icet_image = icetCompositeImage(ubytes,
                                      NULL,
                                      NULL,
                                      NULL,
                                      NULL,
                                      icet_bg_color);
    CHECK_ICET_ERROR();
    GetTimings();
    
    // Only rank 0 has the image
    int rank;
    icetGetIntegerv(ICET_RANK, &rank);
    unsigned char * res = NULL;
    if(rank == 0) 
    {
        res = icetImageGetColorub(m_icet_image);
    }
    delete[] ubytes; 
    return res;
}



//-----------------------------------------------------------------------------
unsigned char *
IceTCompositor::Composite(int width,
                          int height,
                          const unsigned char *color_buffer,
                          const float *depth_buffer,
                          const int   *viewport,
                          const float *bg_color)
{
    icetResetTiles();
    CHECK_ICET_ERROR();

    icetAddTile(0, 0, width, height, 0);
    CHECK_ICET_ERROR();
    
    //best strategy for use with a single tile (i.e., one monitor)
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); 

    CHECK_ICET_ERROR();

    icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);

    CHECK_ICET_ERROR();
    
    // is this necessary?
    IceTFloat icet_bg_color[4] = { bg_color[0],
                                   bg_color[1],
                                   bg_color[2],
                                   bg_color[3] };

    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
    icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);
    
    m_icet_image = icetCompositeImage(color_buffer,
                                      depth_buffer,
                                      viewport,
                                      NULL,
                                      NULL,
                                      icet_bg_color);
    CHECK_ICET_ERROR();
    GetTimings();    

    unsigned char * res= NULL;
    if(m_rank == 0)
    {
        res = icetImageGetColorub(m_icet_image);
    }
    
    return res;
}

unsigned char *
IceTCompositor::Composite(int width,
                          int height,
                          const float *color_buffer,
                          const float *depth_buffer,
                          const int   *viewport,
                          const float *bg_color)
{
    icetResetTiles();
    icetAddTile(0, 0, width, height, 0);
    
    unsigned char * ubytes = ConvertBuffer(color_buffer, width * height * 4);     
    //best strategy for use with a single tile (i.e., one monitor)
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); 
    icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);

    CHECK_ICET_ERROR();
    
    // is this necessary?
    IceTFloat icet_bg_color[4] = { bg_color[0],
                                   bg_color[1],
                                   bg_color[2],
                                   bg_color[3] };

    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
    icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);

    m_icet_image = icetCompositeImage(ubytes,
                                      depth_buffer,
                                      viewport,
                                      NULL,
                                      NULL,
                                      icet_bg_color);

    CHECK_ICET_ERROR();
    GetTimings();    
    // Only rank 0 has the image
    int rank;
    icetGetIntegerv(ICET_RANK, &rank);
    delete[] ubytes;
    if(rank == 0) return icetImageGetColorub(m_icet_image);
    else return NULL;
}


//-----------------------------------------------------------------------------
void
IceTCompositor::GetTimings()
{
  double time;
  icetGetDoublev(ICET_COLLECT_TIME, &time);
  m_log_stream<<"icet_collect_time"<<" "<<time<<"\n";

  icetGetDoublev(ICET_COMPOSITE_TIME, &time);
  m_log_stream<<"icet_composite_time"<<" "<<time<<"\n";

  icetGetDoublev(ICET_BLEND_TIME, &time);
  m_log_stream<<"icet_blend_time"<<" "<<time<<"\n";

  icetGetDoublev(ICET_COMPRESS_TIME, &time);
  m_log_stream<<"icet_COMPRESS_time"<<" "<<time<<"\n";
}

//-----------------------------------------------------------------------------
void
IceTCompositor::Cleanup()
{
    // not sure if we need to do this:
    m_icet_image = icetImageNull();
    icetDestroyContext(m_icet_context);
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



