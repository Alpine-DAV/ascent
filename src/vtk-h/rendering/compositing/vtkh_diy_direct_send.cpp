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
  const int *   m_vis_order;
  const float * m_bg_color;
  Redistribute(const Decomposer &decomposer,
               const int *       vis_order = NULL,
               const float *     bg_color = NULL)
    : m_decomposer(decomposer),
      m_vis_order(vis_order),
      m_bg_color(bg_color)
  {}

  void operator()(void *v_block, const diy::ReduceProxy &proxy) const
  {
    ImageBlock *block = static_cast<ImageBlock*>(v_block);
    //
    // first round we have no incoming. Take the image we have,
    // chop it up into pieces, and send it to the domain resposible
    // for that portion 
    //
    const int rank = proxy.gid();
    const int world_size = m_decomposer.nblocks;
    
    if(proxy.in_link().size() == 0)
    {
      std::map<diy::BlockID,Image> outgoing;

      for(int i = 0; i < world_size; ++i)
      {
        diy::DiscreteBounds sub_image_bounds;
        m_decomposer.fill_bounds(sub_image_bounds, i);
        vtkm::Bounds vtkm_sub_bounds = DIYBoundsToVTKM(sub_image_bounds);
        
        diy::BlockID dest = proxy.out_link().target(i); 
        outgoing[dest].SubsetFrom(block->m_image, vtkm_sub_bounds); 
        std::cout<<outgoing[dest].ToString()<<"\n";
      } //for

      typename std::map<diy::BlockID,Image>::iterator it;
      for(it = outgoing.begin(); it != outgoing.end(); ++it)
      {
        proxy.enqueue(it->first, it->second);
      }
    } // if
    else if(!block->m_image.m_z_buffer_mode)
    {
      // blend images according to vis order
      assert(m_vis_order != NULL);
      assert(m_bg_color != NULL);
      std::vector<Image> incoming(world_size);
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, incoming[gid]); 
        //std::cout<<"rank "<<rank<<" rec "<<incoming[gid].ToString()<<"\n";
      } // for

      const int start = m_vis_order[0];
      for(int i = 1; i < world_size; ++i)
      {
        const int next = m_vis_order[i]; 
        incoming[start].Blend(incoming[next]);
      }

      block->m_image.Swap(incoming[start]);
      block->m_image.CompositeBackground(m_bg_color);
      std::stringstream ss;
      ss<<rank<<"_part.png";
      block->m_image.Save(ss.str());
    } // else if
    else
    {
      /*
      // z buffer compositing
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        Image image;
        int gid = proxy.in_link().target(i).gid;
        proxy.dequeue(gid, image); 
        block
      } // for
      */
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
                                      Image                  &image, 
                                      const int *             vis_order,
                                      const float *           bg_color)
{
  std::stringstream ss;
  diy::DiscreteBounds global_bounds = VTKMBoundsToDIY(image.m_orig_bounds);
  
  // tells diy to use all availible threads
  const int num_threads = -1; 
  const int num_blocks = diy_comm.size(); 
  const int magic_k = 8;

  diy::Master master(diy_comm, num_threads);
  
  // create an assigner with one block per rank
  diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
  AddImageBlock create(master, image);

  const int dims = 2;
  diy::RegularDecomposer<diy::DiscreteBounds> decomposer(dims, global_bounds, num_blocks);
  decomposer.decompose(diy_comm.rank(), assigner, create);
  
  diy::all_to_all(master, 
                  assigner, 
                  Redistribute(decomposer, vis_order, bg_color), 
                  magic_k);

  diy::all_to_all(master,
                  assigner,
                  CollectImages(decomposer),
                  magic_k);
  if(diy_comm.rank() == 0) 
  {
    master.prof.output(m_timing_log);
  }
}

std::string 
DirectSendCompositor::GetTimingString()
{
  std::string res(m_timing_log.str());
  m_timing_log.str("");
  return res;
}

}
