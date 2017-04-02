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
/// file: alpine_diy_collect.hpp
///
//-----------------------------------------------------------------------------
#ifndef ALPINE_DIY_COLLECT_HPP
#define ALPINE_DIY_COLLECT_HPP

#include <diy/master.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include "alpine_diy_image.hpp"

// -- begin alpine:: --
//-----------------------------------------------------------------------------

namespace alpine 
{

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
      Image final_image(block->m_image.m_orig_bounds, block->m_image.m_z_buffer_mode); 
      block->m_image.SubsetTo(final_image);
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        if(gid == collection_rank) 
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

} 
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
