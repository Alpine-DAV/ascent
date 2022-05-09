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
#ifndef APCOMP_DIY_COLLECT_h
#define APCOMP_DIY_COLLECT_h

#include <apcomp/absorption_partial.hpp>
#include <apcomp/emission_partial.hpp>
#include <apcomp/volume_partial.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>

namespace apcomp {
//
// Collect struct sends all data to a single node.
//
template<typename BlockType>
struct Collect
{
  const apcompdiy::RegularDecomposer<apcompdiy::ContinuousBounds> &m_decomposer;

  Collect(const apcompdiy::RegularDecomposer<apcompdiy::ContinuousBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *v_block, const apcompdiy::ReduceProxy &proxy) const
  {
    BlockType *block = static_cast<BlockType*>(v_block);
    //
    // first round we have no incoming. Take the partials we have
    // and sent them to to the right rank
    //
    const int collection_rank = 0;
    if(proxy.in_link().size() == 0 && proxy.gid() != collection_rank)
    {
      int dest_gid = collection_rank;
      apcompdiy::BlockID dest = proxy.out_link().target(dest_gid);
      proxy.enqueue(dest, block->m_partials);

      block->m_partials.clear();

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
        std::vector<typename BlockType::PartialType> incoming_partials;
        proxy.dequeue(gid, incoming_partials);
        const int incoming_size = incoming_partials.size();
        // TODO: make this a std::copy
        for(int j = 0; j < incoming_size; ++j)
        {
          block->m_partials.push_back(incoming_partials[j]);
        }
      } // for
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

  apcompdiy::mpi::communicator world(comm);
  const int dims = 1; // 1D decomp
  apcompdiy::ContinuousBounds global_bounds(dims);
  global_bounds.min[0] = 0;
  global_bounds.max[0] = 1;

  // tells diy to use all availible threads
  const int num_threads = -1;
  const int num_blocks = world.size();
  const int magic_k = 2;

  apcompdiy::Master master(world, num_threads);

  // create an assigner with one block per rank
  apcompdiy::ContiguousAssigner assigner(num_blocks, num_blocks);
  AddBlockType create(master, partials);

  apcompdiy::RegularDecomposer<apcompdiy::ContinuousBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(world.rank(), assigner, create);

  apcompdiy::all_to_all(master, assigner, Collect<Block>(decomposer), magic_k);


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

