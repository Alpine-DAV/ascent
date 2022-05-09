#ifndef APCOMP_DIY_COLLECT_HPP
#define APCOMP_DIY_COLLECT_HPP

#include <diy/master.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include <apcomp/image.hpp>
#include <apcomp/internal/apcomp_diy_image_block.hpp>

namespace apcomp
{

template<typename ImageType>
struct CollectImages
{
  const apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> &m_decomposer;

  CollectImages(const apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *b, const apcompdiy::ReduceProxy &proxy) const
  {
    ImageBlock<ImageType> *block = reinterpret_cast<ImageBlock<ImageType>*>(b);
    //
    // first round we have no incoming. Take the images we have
    // and sent them to to the right rank
    //
    const int collection_rank = 0;
    if(proxy.in_link().size() == 0)
    {

      if(proxy.gid() != collection_rank)
      {
        int dest_gid = collection_rank;
        apcompdiy::BlockID dest = proxy.out_link().target(dest_gid);

        proxy.enqueue(dest, block->m_image);
        block->m_image.Clear();
      }
    } // if
    else if(proxy.gid() == collection_rank)
    {
      ImageType final_image;
      final_image.InitOriginal(block->m_image);
      block->m_image.SubsetTo(final_image);

      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;

        if(gid == collection_rank)
        {
          continue;
        }
        ImageType incoming;
        proxy.dequeue(gid, incoming);
        incoming.SubsetTo(final_image);
      } // for
      block->m_image.Swap(final_image);
    } // else

  } // operator
};

} // namespace apcomp
#endif
