//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
//[Ma6 
// All rights reserved.
// 
// Thi[Ma6s file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// [Ma6
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
// [Ma6
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
//~~~~~~~~~~~~~~~~~~~~~~~~~[Ma6~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///[Ma6
/// file: alpine_diy_compositor.cpp
///
//-----[Ma6------------------------------------------------------------------------

#include "alpine_diy_compositor.hpp"
#include "alpine_config.h"
#include "alpine_logging.hpp"
#include <diy/master.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>

#include <assert.h>
#include <limits> 

// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

void redistribute_pixels(void *b, 
                         const diy::ReduceProxy &proxy,
                         const diy::RegularSwapPartners &partners) 
{
  ImageBlock *block = reinterpret_cast<ImageBlock*>(b);
  unsigned int round = proxy.round();
  Image &image = block->m_image; 
  //if(round == 0) image.Color(proxy.gid());
  //if(proxy.gid() == 0) fmt::print(std::cout, "Round [{}] \n\n\n",round);
  // count the number of incoming pixels
  if(proxy.in_link().size() > 0)
  {
      //fmt::print(std::cout, "Round [{}] recieving\n",round);
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        if(gid == proxy.gid())
        {
          //skip revieving from self since we sent nothing
          continue;
        }
        Image incoming; 
        proxy.dequeue(gid, incoming);
        /*
        fmt::print(std::cout, 
                   "[{}] Recieved {} from [{}] {}\n",
                   proxy.gid(), 
                   incoming.m_depths.size(), 
                   gid,
                   incoming.ToString());
         */
        image.Composite(incoming);
      } // for in links
  } 
  if(proxy.out_link().size() == 0)
  {
    return;
  }
  // do compositing?? intermediate stage?
  const int group_size = proxy.out_link().size(); 
  const int current_dim = partners.dim(round);
  
  const int size = image.m_depths.size(); 
  //create balanced set of ranges for current dim
  int range_length = image.m_bounds.max[current_dim] - image.m_bounds.min[current_dim];
  int base_step = range_length / group_size;
  int rem = range_length % group_size;
  std::vector<int> bucket_sizes(group_size, base_step);
  for(int i  = 0; i < rem; ++i)
  {
    bucket_sizes[i]++;
  }

  int count = 0;
  for(int i  = 0; i < group_size; ++i)
  {
    count += bucket_sizes[i];
  }
  assert(count == range_length);

  std::vector<diy::DiscreteBounds> subset_bounds(group_size, image.m_bounds);  
  int min_pixel = image.m_bounds.min[current_dim];
  for(int i = 0; i < group_size; ++i)
  {
    subset_bounds[i].min[current_dim] = min_pixel; 
    subset_bounds[i].max[current_dim] = min_pixel + bucket_sizes[i];
    min_pixel += bucket_sizes[i];
  }
 
  //debug
  const int size_minus_one = group_size - 1;
  if(group_size > 1)
  {
    for(int i = 1; i < group_size; ++i)
    {
      assert(subset_bounds[i-1].max[current_dim] == subset_bounds[i].min[current_dim]);
    }
  
    assert(subset_bounds[0].min[current_dim] == image.m_bounds.min[current_dim]);
    assert(subset_bounds[group_size-1].max[current_dim] == image.m_bounds.max[current_dim]);
  }
  
  std::vector<Image> out_images(group_size);
  for(int i = 0; i < group_size; ++i)
  {
    out_images[i].SubsetFrom(image, subset_bounds[i]);  
  } //for

  for(int i = 0; i < group_size; ++i)
  {
      if(proxy.out_link().target(i).gid == proxy.gid())
      {
        image.Swap(out_images[i]);
      }
      else
      {
        proxy.enqueue(proxy.out_link().target(i), out_images[i]);
        /*
        fmt::print(std::cout, "[{}] Sent to [{}] {}\n",
                   proxy.gid(), 
                   proxy.out_link().target(i).gid,
                   out_images[i].ToString());*/
      }
  } //for 


} // merge_pixels

struct CollectImages
{
  const diy::RegularDecomposer<diy::DiscreteBounds> &m_decomposer;

  CollectImages(const diy::RegularDecomposer<diy::DiscreteBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *b, const diy::ReduceProxy &proxy) const
  {
    ImageBlock *block = reinterpret_cast<ImageBlock*>(b);
    //
    // first round we have no incoming. Take the images we have
    // and sent them to to the right rank
    //
    const int collection_rank = 0; 
    if(proxy.in_link().size() == 0)
    {
      if(proxy.gid() != collection_rank)
      {
        int dest_gid =  collection_rank;
        diy::BlockID dest = proxy.out_link().target(dest_gid);
        proxy.enqueue(dest, block->m_image);
        block->m_image.Clear();
      }
    } // if
    else if(proxy.gid() == collection_rank)
    {
      Image final_image(block->m_image.m_orig_bounds); 
      block->m_image.SubsetTo(final_image);
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        if(gid == collection_rank ) 
        {
          continue;
        }
        Image incoming;
        proxy.dequeue(gid, incoming); 
        incoming.SubsetTo(final_image);
      } // for 
      block->m_image.Swap(final_image);
    } // else

  } // operator
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
DIYCompositor::Composite()
{
    
    diy::DiscreteBounds global_bounds = m_image.m_orig_bounds;

    // tells diy to use all availible threads
    const int num_threads = -1; 
    const int num_blocks = m_diy_comm.size(); 
    const int magic_k = 64;

    diy::Master master(m_diy_comm, num_threads);

    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
    AddImageBlock create(master, m_image);
    const int num_dims = 2;
    diy::RegularDecomposer<diy::DiscreteBounds> decomposer(num_dims, global_bounds, num_blocks);
    decomposer.decompose(m_rank, assigner, create);
    diy::RegularSwapPartners partners(decomposer, 
                                      magic_k, 
                                      false); // false == distance halving
    if(m_rank == 0)
    {
      for(int i =0; i<m_image.m_depths.size(); ++i)
      {
        float d = m_image.m_depths[i];
        //if(d > 0 && d < 1) std::cout<<" " <<d;
         
      }
    }
    diy::reduce(master,
                assigner,
                partners,
                redistribute_pixels);

    diy::all_to_all(master,
                    assigner,
                    CollectImages(decomposer),
                    magic_k);
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
    m_image.Init(color_buffer,
                 depth_buffer,
                 width,
                 height);

    this->Composite();
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
    this->Composite();

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
}

}; //namespace alpine
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



