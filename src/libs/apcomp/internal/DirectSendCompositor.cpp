#include <apcomp/internal/ImageCompositor.hpp>
#include <apcomp/internal/DirectSendCompositor.hpp>
#include <apcomp/internal/MPICollect.hpp>
#include <apcomp/internal/apcomp_diy_collect.hpp>
#include <apcomp/internal/apcomp_diy_utils.hpp>

#include <diy/master.hpp>
#include <diy/mpi.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>

namespace apcomp
{

struct Redistribute
{
  typedef apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> Decomposer;
  const apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> &m_decomposer;
  Redistribute(const Decomposer &decomposer)
    : m_decomposer(decomposer)
  {}

  void operator()(void *v_block, const apcompdiy::ReduceProxy &proxy) const
  {
    MultiImageBlock *block = static_cast<MultiImageBlock*>(v_block);
    //
    // first round we have no incoming. Take the image we have,
    // chop it up into pieces, and send it to the domain resposible
    // for that portion
    //
    const int world_size = m_decomposer.nblocks;
    const int local_images = block->m_images.size();
    if(proxy.in_link().size() == 0)
    {
      std::map<apcompdiy::BlockID, std::vector<Image>> outgoing;

      for(int i = 0; i < world_size; ++i)
      {
        apcompdiy::DiscreteBounds sub_image_bounds(2);
        m_decomposer.fill_bounds(sub_image_bounds, i);
        Bounds vtkm_sub_bounds = DIYToBounds(sub_image_bounds);

        apcompdiy::BlockID dest = proxy.out_link().target(i);
        outgoing[dest].resize(local_images);

        for(int img = 0;  img < local_images; ++img)
        {
          outgoing[dest][img].SubsetFrom(block->m_images[img], vtkm_sub_bounds);
        }
      } //for

      typename std::map<apcompdiy::BlockID,std::vector<Image>>::iterator it;
      for(it = outgoing.begin(); it != outgoing.end(); ++it)
      {
        proxy.enqueue(it->first, it->second);
      }
    } // if
    else if(block->m_images.at(0).m_composite_order != -1)
    {
      // blend images according to vis order
      std::vector<Image> images;
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {

        std::vector<Image> incoming;
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, incoming);
        const int in_size = incoming.size();
        for(int img = 0; img < in_size; ++img)
        {
          images.emplace_back(incoming[img]);
          //std::cout<<"rank "<<rank<<" rec "<<incoming[img].ToString()<<"\n";
        }
      } // for

      ImageCompositor compositor;
      compositor.OrderedComposite(images);

      block->m_output.Swap(images[0]);
    } // else if
    else if(block->m_images.at(0).m_composite_order == -1 &&
            block->m_images.at(0).HasTransparency())
    {
      std::vector<Image> images;
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {

        std::vector<Image> incoming;
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, incoming);
        const int in_size = incoming.size();
        for(int img = 0; img < in_size; ++img)
        {
          images.emplace_back(incoming[img]);
          //std::cout<<"rank "<<rank<<" rec "<<incoming[img].ToString()<<"\n";
        }
      } // for

      //
      // we have images with a depth buffer and transparency
      //
      ImageCompositor compositor;
      compositor.ZBufferBlend(images);
    }

  } // operator
};

DirectSendCompositor::DirectSendCompositor()
{

}

DirectSendCompositor::~DirectSendCompositor()
{

}

void
DirectSendCompositor::CompositeVolume(apcompdiy::mpi::communicator &diy_comm,
                                      std::vector<Image>     &images)
{
  apcompdiy::DiscreteBounds global_bounds = BoundsToDIY(images.at(0).m_orig_bounds);

  const int num_threads = 1;
  const int num_blocks = diy_comm.size();
  const int magic_k = 8;
  Image sub_image;
  //
  // DIY does not seem to like being called with different block types
  // so we isolate them within separate blocks
  //
  {
    apcompdiy::Master master(
        diy_comm,
        num_threads,
        -1, 0,
        [](void * b){
           MultiImageBlock *block = reinterpret_cast<MultiImageBlock*>(b);
           delete block;
        });

    // create an assigner with one block per rank
    apcompdiy::ContiguousAssigner assigner(num_blocks, num_blocks);

    AddMultiImageBlock create(master, images, sub_image);

    const int dims = 2;
    apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
    decomposer.decompose(diy_comm.rank(), assigner, create);

    apcompdiy::all_to_all(master,
                    assigner,
                    Redistribute(decomposer),
                    magic_k);
  }

  {
    apcompdiy::Master master(
        diy_comm,
        num_threads,
        -1, 0,
        [](void * b){
           ImageBlock<Image> *block = reinterpret_cast<ImageBlock<Image>*>(b);
           delete block;
        });
    apcompdiy::ContiguousAssigner assigner(num_blocks, num_blocks);

    const int dims = 2;
    apcompdiy::RegularDecomposer<apcompdiy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
    AddImageBlock<Image> all_create(master, sub_image);
    decomposer.decompose(diy_comm.rank(), assigner, all_create);
    MPI_Barrier(diy_comm);

    //MPICollect(sub_image,diy_comm);
    apcompdiy::all_to_all(master,
                    assigner,
                    CollectImages<Image>(decomposer),
                    magic_k);
  }

  images.at(0).Swap(sub_image);
}

std::string
DirectSendCompositor::GetTimingString()
{
  std::string res(m_timing_log.str());
  m_timing_log.str("");
  return res;
}

}
