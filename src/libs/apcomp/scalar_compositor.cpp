//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "scalar_compositor.hpp"
#include <apcomp/internal/ScalarImageCompositor.hpp>

#include <assert.h>
#include <algorithm>

#ifdef APCOMP_PARALLEL
#include <mpi.h>
#include <apcomp/apcomp.hpp>
#include <apcomp/internal/RadixKCompositor.hpp>
#include <diy/mpi.hpp>
#endif

namespace apcomp
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
PayloadCompositor::AddImage(ScalarImage &image)
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
    apcomp::ScalarImageCompositor compositor;
    compositor.ZBufferComposite(m_images[0],image);
  }
}

ScalarImage
PayloadCompositor::Composite()
{
  assert(m_images.size() != 0);

  // nothing to do here in serial. Images were composited as
  // they were added to the compositor
#ifdef APCOMP_PARALLEL
  apcompdiy::mpi::communicator diy_comm;
  diy_comm = apcompdiy::mpi::communicator(MPI_Comm_f2c(mpi_comm()));

  assert(m_images.size() == 1);
  RadixKCompositor compositor;
  compositor.CompositeSurface(diy_comm, this->m_images[0]);
#endif
  // Make this a param to avoid the copy?
  return m_images[0];
}


} // namespace apcomp


