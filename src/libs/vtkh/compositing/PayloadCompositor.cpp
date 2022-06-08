#include "PayloadCompositor.hpp"
#include <vtkh/compositing/PayloadImageCompositor.hpp>

#include <assert.h>
#include <algorithm>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/compositing/RadixKCompositor.hpp>
#include <diy/mpi.hpp>
#endif

namespace vtkh
{

PayloadCompositor::PayloadCompositor()
{

}

void
PayloadCompositor::ClearImages()
{
  m_images.clear();
}

void
PayloadCompositor::AddImage(PayloadImage &image)
{
  assert(image.GetNumberOfPixels() != 0);

  if(m_images.size() == 0)
  {
    m_images.push_back(image);
  }
  else
  {
    //
    // Do local composite and keep a single image
    //
    vtkh::PayloadImageCompositor compositor;
    compositor.ZBufferComposite(m_images[0],image);
  }
}

PayloadImage
PayloadCompositor::Composite()
{
  assert(m_images.size() != 0);
  // nothing to do here in serial. Images were composited as
  // they were added to the compositor
#ifdef VTKH_PARALLEL
  vtkhdiy::mpi::communicator diy_comm;
  diy_comm = vtkhdiy::mpi::communicator(MPI_Comm_f2c(GetMPICommHandle()));

  assert(m_images.size() == 1);
  RadixKCompositor compositor;
  compositor.CompositeSurface(diy_comm, this->m_images[0]);
#endif
  // Make this a param to avoid the copy?
  return m_images[0];
}


} // namespace vtkh


