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
#include <sstream>
#include <ascent_config.h>
//-----------------------------------------------------------------------------
///
/// file: ascent_compositor_base.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_COMPOSITOR_BASE_HPP
#define ASCENT_COMPOSITOR_BASE_HPP

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class Compositor
{
public:
     Compositor() {};
     virtual ~Compositor() {};
    
    virtual void      Init(MPI_Comm mpi_comm) = 0;
    
    // composite with given visibility ordering.
    
    virtual unsigned char    *Composite(int                  width,
                                        int                  height,
                                        const unsigned char *color_buffer,
                                        const int           *vis_order,
                                        const float         *bg_color) = 0;

    virtual unsigned char    *Composite(int                  width,
                                        int                  height,
                                        const float         *color_buffer,
                                        const int           *vis_order,
                                        const float         *bg_color) = 0;

    // composite with using a depth buffer.
    
    virtual unsigned char    *Composite(int                  width,
                                        int                  height,
                                        const unsigned char *color_buffer,
                                        const float         *depth_buffer,
                                        const int           *viewport,
                                        const float         *bg_color) = 0;

    virtual unsigned char            *Composite(int                  width,
                                                int                  height,
                                                const float         *color_buffer,
                                                const float         *depth_buffer,
                                                const int           *viewport,
                                                const float         *bg_color) = 0;


    virtual void         Cleanup() = 0;
    
    std::string          GetLogString() 
    { 
        std::string res = m_log_stream.str(); 
        m_log_stream.str("");
        return res;
    }     

    unsigned char * ConvertBuffer(const float *buffer, const int size)
    {
        unsigned char *ubytes = new unsigned char[size];

#ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < size; ++i)
        {
            ubytes[i] = static_cast<unsigned char>(buffer[i] * 255.f);
        }

        return ubytes;
    }

protected:
    std::stringstream m_log_stream;    
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


