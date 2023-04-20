//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_DIY_PARTIAL_REDISTRIBUTE_HPP
#define APCOMP_DIY_PARTIAL_REDISTRIBUTE_HPP

#include <apcomp/apcomp_config.h>

#include "apcomp_diy_partial_blocks.hpp"
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <map>

namespace apcomp {
//
// Redistributes partial composites to the ranks that owns
// that sectoon of the image. Currently, the domain is decomposed
// in 1-D from min_pixel to max_pixel.
//
template<typename BlockType>
struct Redistribute
{
  const apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> &m_decomposer;

  Redistribute(const apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *v_block, const apcompdiy::ReduceProxy &proxy) const
  {
    BlockType *block = static_cast<BlockType*>(v_block);
    //
    // first round we have no incoming. Take the partials we have
    // and sent them to to the right rank
    //
    if(proxy.in_link().size() == 0)
    {
      const int size = block->m_partials.size();
      std::map<apcompdiy::BlockID,std::vector<typename BlockType::PartialType>> outgoing;

      for(int i = 0; i < size; ++i)
      {
        //apcompdiy::DynamicPoint<int,DIY_MAX_DIM> point(block->m_partials[i].m_pixel_id);
        apcompdiy::DynamicPoint<int,DIY_MAX_DIM> point(1);
        point[0] = block->m_partials[i].m_pixel_id;
        int dest_gid = m_decomposer.point_to_gid(point);
        apcompdiy::BlockID dest = proxy.out_link().target(dest_gid);
        outgoing[dest].push_back(block->m_partials[i]);
      } //for

      block->m_partials.clear();


      for(int i = 0; i < proxy.out_link().size(); ++i)
      {
        int dest_gid = proxy.out_link().target(i).gid;
        apcompdiy::BlockID dest = proxy.out_link().target(dest_gid);
        proxy.enqueue(dest, outgoing[dest]);
        //outgoing[dest].clear();
      }

    } // if
    else
    {
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

  apcompdiy::mpi::communicator world(comm);
  const int dims = 1; // we are doing a 1d decomposition
  apcompdiy::DiscreteBounds global_bounds(dims);
  global_bounds.min[0] = domain_min_pixel;
  global_bounds.max[0] = domain_max_pixel;

  // tells diy to use all availible threads
  const int num_threads = 1;
  const int num_blocks = world.size();
  const int magic_k = 2;

  apcompdiy::Master master(world, num_threads);

  // create an assigner with one block per rank
  apcompdiy::ContiguousAssigner assigner(num_blocks, num_blocks);
  AddBlockType create(master, partials);

  apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(world.rank(), assigner, create);
  apcompdiy::all_to_all(master, assigner, Redistribute<Block>(decomposer), magic_k);
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
