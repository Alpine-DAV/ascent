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
/// file: alpine_diy_compositor.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_diy_compositor.hpp"

#include "alpine_logging.hpp"

 
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
    // TODO: cleanup?
}

//-----------------------------------------------------------------------------
void
DIYCompositor::Init(MPI_Comm mpi_comm)
{
    m_mpi_comm    = mpi_comm;
    MPI_Comm_rank(mpi_comm, &m_rank);
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
float *
DIYCompositor::Composite(int            width,
                          int            height,
                          const float   *color_buffer,
                          const int     *vis_order,
                          const float   *bg_color)
{
    return NULL;
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
    
    unsigned char * res= NULL;
    return res;
}

float *
DIYCompositor::Composite(int width,
                          int height,
                          const float *color_buffer,
                          const float *depth_buffer,
                          const int   *viewport,
                          const float *bg_color)
{
    return NULL;
}


//-----------------------------------------------------------------------------
void
DIYCompositor::Cleanup()
{
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



