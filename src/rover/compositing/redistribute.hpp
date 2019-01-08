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
#ifndef rover_compositing_redistribute_h
#define rover_compositing_redistribute_h

#define DIY_PROFILE

#include <compositing/volume_partial.hpp>
#include <compositing/blocks.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <map>
#include <utils/rover_logging.hpp>

namespace rover{
//
// Redistributes partial composites to the ranks that owns
// that sectoon of the image. Currently, the domain is decomposed
// in 1-D from min_pixel to max_pixel.
//
template<typename BlockType>
struct Redistribute
{
  const diy::RegularDecomposer<diy::DiscreteBounds> &m_decomposer;

  Redistribute(const diy::RegularDecomposer<diy::DiscreteBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *v_block, const diy::ReduceProxy &proxy) const
  {
    BlockType *block = static_cast<BlockType*>(v_block);
    //
    // first round we have no incoming. Take the partials we have
    // and sent them to to the right rank
    //
    if(proxy.in_link().size() == 0)
    {
      const int size = block->m_partials.size();
      ROVER_INFO("Processing partials block of size "<<size);
      std::map<diy::BlockID,std::vector<typename BlockType::PartialType>> outgoing;

      for(int i = 0; i < size; ++i)
      {
        diy::Point<int,DIY_MAX_DIM> point;
        point[0] = block->m_partials[i].m_pixel_id;
        int dest_gid = m_decomposer.point_to_gid(point);
        diy::BlockID dest = proxy.out_link().target(dest_gid);
        outgoing[dest].push_back(block->m_partials[i]);
      } //for

      block->m_partials.clear();

      ROVER_INFO("out setup ");

      for(int i = 0; i < proxy.out_link().size(); ++i)
      {
        int dest_gid = proxy.out_link().target(i).gid;
        diy::BlockID dest = proxy.out_link().target(dest_gid);
        proxy.enqueue(dest, outgoing[dest]);
        //outgoing[dest].clear();
      }

    } // if
    else
    {
      ROVER_INFO("getting "<<proxy.in_link().size()<<" blocks");
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        std::vector<typename BlockType::PartialType> incoming_partials;
        ROVER_INFO("dequing from "<<gid);
        proxy.dequeue(gid, incoming_partials);
        const int incoming_size = incoming_partials.size();
        ROVER_INFO("Incoming size "<<incoming_size<<" from "<<gid);
        // TODO: make this a std::copy
        for(int j = 0; j < incoming_size; ++j)
        {
          block->m_partials.push_back(incoming_partials[j]);
        }
      } // for

    } // else
    MPI_Barrier(MPI_COMM_WORLD); //HACK
  } // operator
};


template<typename AddBlockType>
void redistribute_detail(std::vector<typename AddBlockType::PartialType> &partials,
                         MPI_Comm comm,
                         const int &domain_min_pixel,
                         const int &domain_max_pixel)
{
  typedef typename AddBlockType::Block Block;

  diy::mpi::communicator world(comm);
  diy::DiscreteBounds global_bounds;
  global_bounds.min[0] = domain_min_pixel;
  global_bounds.max[0] = domain_max_pixel;

  // tells diy to use all availible threads
  const int num_threads = 1;
  const int num_blocks = world.size();
  const int magic_k = 2;

  diy::Master master(world, num_threads);

  // create an assigner with one block per rank
  diy::ContiguousAssigner assigner(num_blocks, num_blocks);
  AddBlockType create(master, partials);

  const int dims = 1;
  diy::RegularDecomposer<diy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(world.rank(), assigner, create);
  diy::all_to_all(master, assigner, Redistribute<Block>(decomposer), magic_k);
}

//
// Define a default template that cannot be instantiated
//
template<typename T>
void redistribute(std::vector<T> &partials,
                  MPI_Comm comm,
                  const int &domain_min_pixel,
                  const int &domain_max_pixel);
// ----------------------------- VolumePartial Specialization------------------------------------------
template<>
void redistribute<VolumePartial<float>>(std::vector<VolumePartial<float>> &partials,
                                                                           MPI_Comm comm,
                                                                           const int &domain_min_pixel,
                                                                           const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<VolumeBlock<float>>>(partials,
                                                    comm,
                                                    domain_min_pixel,
                                                    domain_max_pixel);
}

template<>
void redistribute<VolumePartial<double>>(std::vector<VolumePartial<double>> &partials,
                                                                             MPI_Comm comm,
                                                                             const int &domain_min_pixel,
                                                                             const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<VolumeBlock<double>>>(partials,
                                                     comm,
                                                     domain_min_pixel,
                                                     domain_max_pixel);
}

// ----------------------------- AbsorpPartial Specialization------------------------------------------
template<>
void redistribute<AbsorptionPartial<double>>(std::vector<AbsorptionPartial<double>> &partials,
                                             MPI_Comm comm,
                                             const int &domain_min_pixel,
                                             const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<AbsorptionBlock<double>>>(partials,
                                                         comm,
                                                         domain_min_pixel,
                                                         domain_max_pixel);
}

template<>
void redistribute<AbsorptionPartial<float>>(std::vector<AbsorptionPartial<float>> &partials,
                                            MPI_Comm comm,
                                            const int &domain_min_pixel,
                                            const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<AbsorptionBlock<float>>>(partials,
                                                        comm,
                                                        domain_min_pixel,
                                                        domain_max_pixel);
}

// ----------------------------- EmissPartial Specialization------------------------------------------
template<>
void redistribute<EmissionPartial<double>>(std::vector<EmissionPartial<double>> &partials,
                                          MPI_Comm comm,
                                          const int &domain_min_pixel,
                                          const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<EmissionBlock<double>>>(partials,
                                                       comm,
                                                       domain_min_pixel,
                                                       domain_max_pixel);
}

template<>
void redistribute<EmissionPartial<float>>(std::vector<EmissionPartial<float>> &partials,
                                            MPI_Comm comm,
                                            const int &domain_min_pixel,
                                            const int &domain_max_pixel)
{
  redistribute_detail<AddBlock<EmissionBlock<float>>>(partials,
                                                      comm,
                                                      domain_min_pixel,
                                                      domain_max_pixel);
}

} //namespace rover

#endif
