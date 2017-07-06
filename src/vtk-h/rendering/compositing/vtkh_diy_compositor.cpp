#include "vtkh_diy_compositor.hpp"
//#include "alpine_config.h"
#include "alpine_logging.hpp"
#include <vtkh.hpp>
#include "vtkh_diy_direct_send.hpp"
#include "vtkh_diy_radix_k.hpp"
#include <diy/mpi.hpp>

#include <assert.h>
#include <limits> 

namespace vtkh 
{
DIYCompositor::DIYCompositor()
: m_rank(0)
{
    m_diy_comm = diy::mpi::communicator(VTKH::GetMPIComm());
    m_rank = m_diy_comm.rank();
}
  
DIYCompositor::~DIYCompositor()
{
}

void 
DIYCompositor::CompositeZBufferSurface()
{
  assert(m_images.size() == 1);
  RadixKCompositor compositor;

  compositor.CompositeSurface(m_diy_comm, this->m_images[0]);

  m_log_stream<<compositor.GetTimingString();

}

void 
DIYCompositor::CompositeZBufferBlend()
{
  assert("this is not implemented yet" == "error");  
}

void 
DIYCompositor::CompositeVisOrder()
{
  assert(m_images.size() != 0);
  DirectSendCompositor compositor;
  compositor.CompositeVolume(m_diy_comm, this->m_images, m_background_color);
}

void
DIYCompositor::Cleanup()
{

}

}; //namespace vtkh 



