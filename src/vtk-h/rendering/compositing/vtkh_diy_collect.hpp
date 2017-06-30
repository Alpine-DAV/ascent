#ifndef VTKH_DIY_COLLECT_HPP
#define VTKH_DIY_COLLECT_HPP

#include <diy/master.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include <rendering/vtkh_image.hpp>
#include <rendering/compositing/vtkh_diy_image_block.hpp>

namespace vtkh 
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

} // namespace vtkh
#endif
