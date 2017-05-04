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
//   sp[Ma6ecific prior written permission.
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
/// file: alpine_diy_radix_k.cpp
///
//-----------------------------------------------------------------------------
#include "alpine_diy_radix_k.hpp"
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

void reduce_images(void *b, 
                   const diy::ReduceProxy &proxy,
                   const diy::RegularSwapPartners &partners) 
{
  ImageBlock *block = reinterpret_cast<ImageBlock*>(b);
  unsigned int round = proxy.round();
  Image &image = block->m_image; 
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
      }
  } //for 

} // reduce images

RadixKCompositor::RadixKCompositor()
{

}

RadixKCompositor::~RadixKCompositor()
{

}

void
RadixKCompositor::CompositeSurface(diy::mpi::communicator &diy_comm, Image &image)
{
  
    diy::DiscreteBounds global_bounds = image.m_orig_bounds;

    // tells diy to use all availible threads
    const int num_threads = -1; 
    const int num_blocks = diy_comm.size(); 
    const int magic_k = 8;

    diy::Master master(diy_comm, num_threads);

    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
    AddImageBlock create(master, image);
    const int num_dims = 2;
    diy::RegularDecomposer<diy::DiscreteBounds> decomposer(num_dims, global_bounds, num_blocks);
    decomposer.decompose(diy_comm.rank(), assigner, create);
    diy::RegularSwapPartners partners(decomposer, 
                                      magic_k, 
                                      false); // false == distance halving
    diy::reduce(master,
                assigner,
                partners,
                reduce_images);


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
RadixKCompositor::GetTimingString()
{
  std::string res(m_timing_log.str());
  m_timing_log.str("");
  return res;
}

}
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------
