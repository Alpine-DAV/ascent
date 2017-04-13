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
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistribu[Ma6tions in binary form must reproduce the above copyright notice,
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
/// file: alpine_diy_direct_send.cpp
///
//-----------------------------------------------------------------------------
#include "alpine_diy_direct_send.hpp"
#include "alpine_diy_collect.hpp"

#include <diy/master.hpp>
#include <diy/mpi.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>

// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine 
{

struct Redistribute
{
  typedef diy::RegularDecomposer<diy::DiscreteBounds> Decomposer;
  const diy::RegularDecomposer<diy::DiscreteBounds> &m_decomposer;
  const int *   m_vis_order;
  const float * m_bg_color;
  Redistribute(const Decomposer &decomposer,
               const int *       vis_order = NULL,
               const float *     bg_color = NULL)
    : m_decomposer(decomposer),
      m_vis_order(vis_order),
      m_bg_color(bg_color)
  {}

  void operator()(void *v_block, const diy::ReduceProxy &proxy) const
  {
    ImageBlock *block = static_cast<ImageBlock*>(v_block);
    //
    // first round we have no incoming. Take the image we have,
    // chop it up into pieces, and send it to the domain resposible
    // for that portion 
    //
    const int rank = proxy.gid();
    const int world_size = m_decomposer.nblocks;
    
    if(proxy.in_link().size() == 0)
    {
      std::map<diy::BlockID,Image> outgoing;

      for(int i = 0; i < world_size; ++i)
      {
        diy::DiscreteBounds sub_image_bounds;
        m_decomposer.fill_bounds(sub_image_bounds, i);
        
        diy::BlockID dest = proxy.out_link().target(i); 
        outgoing[dest].SubsetFrom(block->m_image, sub_image_bounds); 
        std::cout<<outgoing[dest].ToString()<<"\n";
      } //for

      typename std::map<diy::BlockID,Image>::iterator it;
      for(it = outgoing.begin(); it != outgoing.end(); ++it)
      {
        proxy.enqueue(it->first, it->second);
      }
    } // if
    else if(!block->m_image.m_z_buffer_mode)
    {
      // blend images according to vis order
      assert(m_vis_order != NULL);
      assert(m_bg_color != NULL);
      std::vector<Image> incoming(world_size);
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, incoming[gid]); 
        //std::cout<<"rank "<<rank<<" rec "<<incoming[gid].ToString()<<"\n";
      } // for

      const int start = m_vis_order[0];
      for(int i = 1; i < world_size; ++i)
      {
        const int next = m_vis_order[i]; 
        incoming[start].Blend(incoming[next]);
      }

      block->m_image.Swap(incoming[start]);
      block->m_image.CompositeBackground(m_bg_color);
      std::stringstream ss;
      ss<<rank<<"_part.png";
      block->m_image.Save(ss.str());
    } // else if
    else
    {
      /*
      // z buffer compositing
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        Image image;
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, image); 
        block
      } // for
      */
    }

  } // operator
};
DirectSendCompositor::DirectSendCompositor()
{

}

DirectSendCompositor::~DirectSendCompositor()
{

}

void
DirectSendCompositor::CompositeVolume(diy::mpi::communicator &diy_comm, 
                                      Image                  &image, 
                                      const int *             vis_order,
                                      const float *           bg_color)
{
  std::stringstream ss;
  diy::DiscreteBounds global_bounds = image.m_orig_bounds;;
  
  // tells diy to use all availible threads
  const int num_threads = -1; 
  const int num_blocks = diy_comm.size(); 
  const int magic_k = 8;

  diy::Master master(diy_comm, num_threads);
  
  // create an assigner with one block per rank
  diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
  AddImageBlock create(master, image);

  const int dims = 2;
  diy::RegularDecomposer<diy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(diy_comm.rank(), assigner, create);
  
  diy::all_to_all(master, 
                  assigner, 
                  Redistribute(decomposer, vis_order, bg_color), 
                  magic_k);

  diy::all_to_all(master,
                  assigner,
                  CollectImages(decomposer),
                  magic_k);
  if(diy_comm.rank() == 0) 
  {
    master.prof.output(m_timing_log);
  }
}

std::string 
DirectSendCompositor::GetTimingString()
{
  std::string res(m_timing_log.str());
  m_timing_log.str("");
  return res;
}

}
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------
