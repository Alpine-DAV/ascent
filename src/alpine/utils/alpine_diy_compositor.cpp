//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// Thi[Ma6s file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
// 
// Redi[Ma6stribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redist[Ma6ributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistribu[Ma6tions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   sp[Ma6ecific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY [Ma6EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S[Ma6. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOS[Ma6S OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF TH[Ma6E USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: alpine_diy_compositor.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_diy_compositor.hpp"
#include "alpine_config.h"
#include "alpine_logging.hpp"
#include "diy_compositing/alpine_diy_direct_send.hpp"
#include "diy_compositing/alpine_diy_radix_k.hpp"
#include <diy/mpi.hpp>

#include <assert.h>
#include <limits> 

// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{
//-----------------------------------------------------------------------------
DIYCompositor::DIYCompositor()
: m_rank(0)
{}
  
//-----------------------------------------------------------------------------
DIYCompositor::~DIYCompositor()
{
}

//-----------------------------------------------------------------------------
void
DIYCompositor::Init(MPI_Comm mpi_comm)
{
    m_diy_comm = diy::mpi::communicator(mpi_comm);
    m_rank = m_diy_comm.rank();
}

//-----------------------------------------------------------------------------
unsigned char *
DIYCompositor::Composite(int            width,
                         int            height,
                         const unsigned char *color_buffer,
                         const int           *vis_order,
                         const float         *bg_color)
{
    return NULL;
}

//-----------------------------------------------------------------------------
unsigned char*
DIYCompositor::Composite(int            width,
                         int            height,
                         const float   *color_buffer,
                         const int     *vis_order,
                         const float   *bg_color)
{
    m_image.Init(color_buffer,
                 NULL,
                 width,
                 height);

    DirectSendCompositor compositor;
    compositor.CompositeVolume(m_diy_comm, m_image, vis_order);
    m_image.m_orig_rank = m_rank;

    if(m_rank == 0)
    {
      return &m_image.m_pixels[0];
    }
    else
    {
      return NULL;
    }
}



//-----------------------------------------------------------------------------
unsigned char *
DIYCompositor::Composite(int width,
                         int height,
                         const unsigned char *color_buffer,
                         const float *depth_buffer,
                         const int   *viewport,
                         const float *bg_color)
{
    m_image.Init(color_buffer,
                 depth_buffer,
                 width,
                 height);

    RadixKCompositor compositor;
    compositor.CompositeSurface(m_diy_comm, m_image);

    if(m_rank == 0)
    {
      return &m_image.m_pixels[0];
    }
    else
    {
      return NULL;
    }
}

unsigned char*
DIYCompositor::Composite(int width,
                         int height,
                         const float *color_buffer,
                         const float *depth_buffer,
                         const int   *viewport,
                         const float *bg_color)
{
    m_image.Init(color_buffer,
                 depth_buffer,
                 width,
                 height);

    RadixKCompositor compositor;
    compositor.CompositeSurface(m_diy_comm, m_image);

    if(m_rank == 0)
    {
      return &m_image.m_pixels[0];
    }
    else
    {
      return NULL;
    }
}


//-----------------------------------------------------------------------------
void
DIYCompositor::Cleanup()
{
  if(m_rank == 0)
  {
    std::ofstream log_file;
    log_file.open("composite_timings.log");
    log_file<<m_timing_log.str();
    log_file.close();
  }
}

}; //namespace alpine
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



