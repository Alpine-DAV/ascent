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
#include <diy/master.hpp>

 
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

struct PixelBlock
{
    std::vector<Pixel>  &m_pixels;
    PixelBlock(std::vector<Pixel> &pixels)
      : m_pixels(pixels)
    {}
};

struct AddPixelBlock
{
  std::vector<Pixel> &m_pixels;
  const diy::Master &m_master;

  AddPixelBlock(diy::Master &master,std::vector<Pixel> &pixels)
    : m_master(master), m_pixels(pixels)
  {}
  template<typename BoundsType, typename LinkType>                 
  void operator()(int gid,
                  const BoundsType &local_bounds,
                  const BoundsType &local_with_ghost_bounds,
                  const BoundsType &domain_bounds,
                  const LinkType &link) const
  {
    PixelBlock *block = new PixelBlock(m_pixels);
    LinkType *linked = new LinkType(link);
    diy::Master& master = const_cast<diy::Master&>(m_master);
    int lid = master.add(gid, block, linked);
  }
}; 
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
    std::vector<Pixel> pixels;
    
    this->Extract(color_buffer,
                  depth_buffer,
                  width,
                  height,
                  pixels);

    diy::DiscreteBounds global_bounds;
    global_bounds.min[0] = 0;
    global_bounds.min[1] = 0;
    global_bounds.max[0] = width;
    global_bounds.max[1] = height;

    // tells diy to use all availible threads
    const int num_threads = -1; 
    const int num_blocks = m_diy_comm.size(); 
    const int magic_k = 64;

    diy::Master master(m_diy_comm, num_threads);

    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
    AddPixelBlock create(master, pixels);
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
    std::vector<Pixel> pixels;
    
    this->Extract(color_buffer,
                  depth_buffer,
                  width, height,
                  pixels);

    diy::DiscreteBounds global_bounds;
    global_bounds.min[0] = 0;
    global_bounds.min[1] = 0;
    global_bounds.max[0] = width;
    global_bounds.max[1] = height;

    // tells diy to use all availible threads
    const int num_threads = -1; 
    const int num_blocks = m_diy_comm.size(); 
    const int magic_k = 64;

    diy::Master master(m_diy_comm, num_threads);

    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
    AddPixelBlock create(master, pixels);
    return NULL;
}


//-----------------------------------------------------------------------------
void
DIYCompositor::Cleanup()
{
}

//-----------------------------------------------------------------------------
template<typename T>
void 
DIYCompositor::Extract(const T            *color_buffer,
                       const float        *depth_buffer,
                       int                 width,
                       int                 height,
                       std::vector<Pixel> &pixels)
{
   const int size = width * height;
#ifdef ALPINE_USE_OPENMP
   std::cout<<"Parallel path\n";

   std::vector<Pixel> all_pixels(size);
   std::vector<char>  flags(size);
   #pragma omp parrallel for
   for(int i = 0; i < size; ++i)
   {
      float depth = depth_buffer[i];
      char flag = 0;
      if(depth >= 0 || depth <= 1) flag = 1;
      flags[i] = flag;
   }

   #pragma omp parrallel for
   for(int i = 0; i < size; ++i)
   {
      float depth = depth_buffer[i];
      char flag = 0;
      if(depth >= 0 || depth < 1) flag = 1;
      flags[i] = flag;
   }

   int real_size = 0;
   #pragma omp parallel for reduction(sum:real_size)
   for(int i = 0; i < size; ++i)
   {
     real_size += static_cast<int>(flags[i]);
   }

   pixels.resize(real_size);
   std::vector<int> offsets(size);
   if(size > 0)
   {
      offsets[0] = 0;
   }
   // scan
   for(int i = 1; i < size; ++i)
   {
      offsets[i] = flags[i-1] + offsets[i-1];
   }

   #pragma omp parrallel for
   for(int i = 0; i < size; ++i)
   {
      if(flags[i] == 0) continue;
      const int offset = offsets[i];
      const int color_offset = offset * 4;    
      pixels[i].SetRGBA(color_buffer + color_offset);
      pixels[i].m_depth = depth_buffer[offset];
      pixels[i].m_pixel_id = i;
   }
#else
  std::cout<<"NON-parallel path\n";
  for(int i = 0; i < size; ++i)
  {
     Pixel pixel;
     const int offset = i * 4;    
     pixel.m_depth = depth_buffer[i];
     if(pixel.m_depth < 0 || pixel.m_depth > 1) continue; 
     pixel.SetRGBA(color_buffer + offset);
     pixel.m_pixel_id = i;
     pixels.push_back(pixel);
  }
#endif
  std::cout<<"Initial size "<<size<<" output size = "<<pixels.size()<<"\n";
}

};
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



