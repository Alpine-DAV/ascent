#ifndef VTKH_DIY_COLLECT_HPP
#define VTKH_DIY_COLLECT_HPP

#include <diy/master.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include <vtkh/compositing/Image.hpp>
#include <vtkh/compositing/vtkh_diy_image_block.hpp>

namespace vtkh
{

template<typename ImageType>
struct CollectImages
{
  const vtkhdiy::RegularDecomposer<vtkhdiy::DiscreteBounds> &m_decomposer;

  CollectImages(const vtkhdiy::RegularDecomposer<vtkhdiy::DiscreteBounds> &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *b, const vtkhdiy::ReduceProxy &proxy) const
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
        vtkhdiy::BlockID dest = proxy.out_link().target(dest_gid);

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

} // namespace vtkh
#endif
