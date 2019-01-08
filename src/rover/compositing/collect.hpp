//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-749865
//
// All rights reserved.
//
// This file is part of Rover.
//
// Please also read rover/LICENSE
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
#ifndef rover_compositing_collect_h
#define rover_compositing_collect_h

#include <compositing/absorption_partial.hpp>
#include <compositing/volume_partial.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <utils/rover_logging.hpp>

namespace rover {
//
// Collect struct sends all data to a single node.
//
template<typename BlockType>
struct Collect
{
  const diy::RegularDecomposer<diy::ContinuousBounds> &m_decomposer;

  Collect(const diy::RegularDecomposer<diy::ContinuousBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *v_block, const diy::ReduceProxy &proxy) const
  {
    BlockType *block = static_cast<BlockType*>(v_block);
    //
    // first round we have no incoming. Take the partials we have
    // and sent them to to the right rank
    //
    ROVER_INFO("Collect: in link "<<proxy.in_link().size()<<" gid "<<proxy.gid());
    const int collection_rank = 0;
    if(proxy.in_link().size() == 0 && proxy.gid() != collection_rank)
    {
      ROVER_INFO("Collect sending partials vector "<<block->m_partials.size());
      int dest_gid = collection_rank;
      diy::BlockID dest = proxy.out_link().target(dest_gid);
      proxy.enqueue(dest, block->m_partials);

      block->m_partials.clear();

      ROVER_INFO("Collect enqueud partials vector "<<block->m_partials.size());
    } // if
    else if(proxy.gid() == collection_rank)
    {

      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        if(gid == collection_rank)
        {
          continue;
        }
        //TODO: leave the paritals that start here, here
        ROVER_INFO("dequeuing from "<<gid);
        std::vector<typename BlockType::PartialType> incoming_partials;
        proxy.dequeue(gid, incoming_partials);
        const int incoming_size = incoming_partials.size();
        ROVER_INFO("dequeuing "<<incoming_size);
        // TODO: make this a std::copy
        for(int j = 0; j < incoming_size; ++j)
        {
          block->m_partials.push_back(incoming_partials[j]);
        }
      } // for
      ROVER_INFO("Collect: done processing incoming");
    } // else

  } // operator
};

//
// collect uses the all-to-all construct to perform a gather to
// the root rank. All other ranks will have no data
//
template<typename AddBlockType>
void collect_detail(std::vector<typename AddBlockType::PartialType> &partials,
                    MPI_Comm comm)
{
  typedef typename AddBlockType::Block Block;

  diy::mpi::communicator world(comm);
  diy::ContinuousBounds global_bounds;
  global_bounds.min[0] = 0;
  global_bounds.max[0] = 1;

  // tells diy to use all availible threads
  const int num_threads = -1;
  const int num_blocks = world.size();
  const int magic_k = 2;

  diy::Master master(world, num_threads);

  // create an assigner with one block per rank
  diy::ContiguousAssigner assigner(num_blocks, num_blocks);
  AddBlockType create(master, partials);

  const int dims = 1;
  diy::RegularDecomposer<diy::ContinuousBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(world.rank(), assigner, create);

  diy::all_to_all(master, assigner, Collect<Block>(decomposer), magic_k);

  ROVER_INFO("Collect ending size: "<<partials.size()<<"\n");

}

template<typename T>
void collect(std::vector<T> &partials,
             MPI_Comm comm);

template<>
void collect<VolumePartial<float>>(std::vector<VolumePartial<float>> &partials,
                                  MPI_Comm comm)
{
  collect_detail<AddBlock<VolumeBlock<float>>>(partials, comm);
}

template<>
void collect<VolumePartial<double>>(std::vector<VolumePartial<double>> &partials,
                                   MPI_Comm comm)
{
  collect_detail<AddBlock<VolumeBlock<double>>>(partials, comm);
}

template<>
void collect<AbsorptionPartial<double>>(std::vector<AbsorptionPartial<double>> &partials,
                                        MPI_Comm comm)
{
  collect_detail<AddBlock<AbsorptionBlock<double>>>(partials, comm);
}

template<>
void collect<AbsorptionPartial<float>>(std::vector<AbsorptionPartial<float>> &partials,
                                       MPI_Comm comm)
{
  collect_detail<AddBlock<AbsorptionBlock<float>>>(partials, comm);
}

template<>
void collect<EmissionPartial<double>>(std::vector<EmissionPartial<double>> &partials,
                                      MPI_Comm comm)
{
  collect_detail<AddBlock<EmissionBlock<double>>>(partials, comm);
}

template<>
void collect<EmissionPartial<float>>(std::vector<EmissionPartial<float>> &partials,
                                     MPI_Comm comm)
{
  collect_detail<AddBlock<EmissionBlock<float>>>(partials, comm);
}

} // namespace rover

#endif

