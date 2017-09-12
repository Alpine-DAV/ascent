#include <rendering/vtkh_image_compositor.hpp>
#include "vtkh_diy_direct_send.hpp"
#include "vtkh_diy_collect.hpp"
#include "vtkh_diy_utils.hpp"

#include <diy/master.hpp>
#include <diy/mpi.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>

namespace vtkh 
{

struct Redistribute
{
  typedef diy::RegularDecomposer<diy::DiscreteBounds> Decomposer;
  const diy::RegularDecomposer<diy::DiscreteBounds> &m_decomposer;
  const float * m_bg_color;
  Redistribute(const Decomposer &decomposer,
               const float *     bg_color = NULL)
    : m_decomposer(decomposer),
      m_bg_color(bg_color)
  {}

  void operator()(void *v_block, const diy::ReduceProxy &proxy) const
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
      std::map<diy::BlockID, std::vector<Image>> outgoing;
      
      for(int i = 0; i < world_size; ++i)
      {
        diy::DiscreteBounds sub_image_bounds;
        m_decomposer.fill_bounds(sub_image_bounds, i);
        vtkm::Bounds vtkm_sub_bounds = DIYBoundsToVTKM(sub_image_bounds);

        diy::BlockID dest = proxy.out_link().target(i); 
        outgoing[dest].resize(local_images); 

        for(int img = 0;  img < local_images; ++img) 
        {
          outgoing[dest][img].SubsetFrom(block->m_images[img], vtkm_sub_bounds); 
        }
      } //for

      typename std::map<diy::BlockID,std::vector<Image>>::iterator it;
      for(it = outgoing.begin(); it != outgoing.end(); ++it)
      {
        proxy.enqueue(it->first, it->second);
      }
    } // if
    else if(!block->m_images.at(0).m_z_buffer_mode)
    {
      // blend images according to vis order
      assert(m_bg_color != NULL);
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
      block->m_output.CompositeBackground(m_bg_color);
    } // else if
    else if(block->m_images.at(0).m_z_buffer_mode &&
            block->m_images.at(0).HasTransparency())
    {
      assert(m_bg_color != NULL);
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
DirectSendCompositor::CompositeVolume(diy::mpi::communicator &diy_comm, 
                                      std::vector<Image>     &images, 
                                      const float *           bg_color)
{
  diy::DiscreteBounds global_bounds = VTKMBoundsToDIY(images.at(0).m_orig_bounds);
  
  const int num_threads = 1; 
  const int num_blocks = diy_comm.size(); 
  const int magic_k = 8;
  Image sub_image;
  //
  // DIY does not seem to like being called with different block types
  // so we isolate them within separate blocks
  //
  {
    diy::Master master(diy_comm, num_threads);
    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 

    AddMultiImageBlock create(master, images, sub_image);

    const int dims = 2;
    diy::RegularDecomposer<diy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
    decomposer.decompose(diy_comm.rank(), assigner, create);
    
    diy::all_to_all(master, 
                    assigner, 
                    Redistribute(decomposer, bg_color), 
                    magic_k);
  }  

  {
    diy::Master master(diy_comm, num_threads);
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 

    const int dims = 2;
    diy::RegularDecomposer<diy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
    AddImageBlock all_create(master, sub_image);
    decomposer.decompose(diy_comm.rank(), assigner, all_create);
    MPI_Barrier(MPI_COMM_WORLD); 

    diy::all_to_all(master,
                    assigner,
                    CollectImages(decomposer),
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
