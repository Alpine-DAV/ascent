//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_DIY_COLLECT_HPP
#define APCOMP_DIY_COLLECT_HPP

#include <apcomp/apcomp_config.h>

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

