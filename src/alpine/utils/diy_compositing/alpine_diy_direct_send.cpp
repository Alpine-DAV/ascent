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

  Redistribute(const Decomposer &decomposer)
    : m_decomposer(decomposer)
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
        if(i == rank) 
        {
          // don't send to self
          continue;
        }
        diy::DiscreteBounds sub_image_bounds;
        m_decomposer.fill_bounds(sub_image_bounds, i);
        
        diy::BlockID dest = proxy.out_link().target(i); 
        outgoing[dest].SubsetFrom(block->m_image, sub_image_bounds); 
        std::cout<<outgoing[dest].ToString()<<"\n";
      } //for
     /* 
      typename std::map<diy::BlockID,std::vector<typename BlockType::PartialType>>::iterator it;
      for( it = outgoing.begin(); it != outgoing.end(); ++it)
      {
        proxy.enqueue(it->first, it->second);
        it->second.clear();
      }*/
    } // if
    else
    {
      /*
      size_t total = 0;
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        std::vector<typename BlockType::PartialType> incoming_partials;
        proxy.dequeue(gid, incoming_partials); 
        const int incoming_size = incoming_partials.size();
        // TODO: make this a std::copy
        for(int j = 0; j < incoming_size; ++j)
        {
          block->m_partials.push_back(incoming_partials[j]);
        }
      } // for
     */ 
    } // else

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
                                      const int *             vis_order)
{
  
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
  
  diy::all_to_all(master, assigner, Redistribute(decomposer), magic_k);
}

}
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------
