//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "compositor.hpp"
#include <apcomp/internal/ImageCompositor.hpp>

#include <assert.h>
#include <algorithm>

#ifdef APCOMP_PARALLEL
#include <mpi.h>
#include <diy/mpi.hpp>
#include <apcomp/apcomp.hpp>
#include <apcomp/internal/DirectSendCompositor.hpp>
#include <apcomp/internal/RadixKCompositor.hpp>
#endif

namespace apcomp
{

Compositor::Compositor()
  : m_composite_mode(Z_BUFFER_SURFACE_GL)
{

}

Compositor::~Compositor()
{

}

void
Compositor::SetCompositeMode(CompositeMode composite_mode)
{
  // assure we don't have mixed image types
  assert(m_images.size() == 0);
  m_composite_mode = composite_mode;
}

void
Compositor::ClearImages()
{
  m_images.clear();
}

void
Compositor::AddImage(const unsigned char *color_buffer,
                     const float *        depth_buffer,
                     const int            width,
                     const int            height)
{
  assert(m_composite_mode != VIS_ORDER_BLEND);
  assert(depth_buffer != NULL);
  Image image;
  bool gl_depth = m_composite_mode != Z_BUFFER_SURFACE_WORLD;
  if(m_images.size() == 0)
  {
    m_images.push_back(image);
    m_images[0].Init(color_buffer,
                     depth_buffer,
                     width,
                     height,
                     gl_depth);
    //m_images[0].Save("first.png");
  }
  else if(m_composite_mode == Z_BUFFER_SURFACE_GL ||
          m_composite_mode == Z_BUFFER_SURFACE_WORLD)
  {
    //
    // Do local composite and keep a single image
    //
    image.Init(color_buffer,
               depth_buffer,
               width,
               height,
               gl_depth);
    apcomp::ImageCompositor compositor;
    compositor.ZBufferComposite(m_images[0],image);
  }
  else
  {
    const size_t image_index = m_images.size();
    m_images.push_back(image);
    m_images[image_index].Init(color_buffer,
                               depth_buffer,
                               width,
                               height);
  }

}

void
Compositor::AddImage(const float *color_buffer,
                     const float *depth_buffer,
                     const int    width,
                     const int    height)
{
  assert(m_composite_mode != VIS_ORDER_BLEND);
  assert(depth_buffer != NULL);
  Image image;

  bool gl_depth = m_composite_mode != Z_BUFFER_SURFACE_WORLD;

  if(m_images.size() == 0)
  {
    m_images.push_back(image);
    m_images[0].Init(color_buffer,
                     depth_buffer,
                     width,
                     height,
                     gl_depth);
  }
  else if(m_composite_mode == Z_BUFFER_SURFACE_GL ||
          m_composite_mode == Z_BUFFER_SURFACE_WORLD)
  {
    //
    // Do local composite and keep a single image
    //
    image.Init(color_buffer,
               depth_buffer,
               width,
               height,
               gl_depth);

    apcomp::ImageCompositor compositor;
    compositor.ZBufferComposite(m_images[0],image);
  }
  else
  {
    const size_t image_index = m_images.size();
    m_images.push_back(image);
    m_images[image_index].Init(color_buffer,
                               depth_buffer,
                               width,
                               height);
  }

}

void
Compositor::AddImage(const unsigned char *color_buffer,
                     const float         *depth_buffer,
                     const int            width,
                     const int            height,
                     const int            vis_order)
{
  assert(m_composite_mode == VIS_ORDER_BLEND);
  Image image;
  const size_t image_index = m_images.size();
  m_images.push_back(image);
  bool gl_depth = false;
  m_images[image_index].Init(color_buffer,
                             depth_buffer,
                             width,
                             height,
                             gl_depth,
                             vis_order);
}

void
Compositor::AddImage(const float *color_buffer,
                     const float *depth_buffer,
                     const int    width,
                     const int    height,
                     const int    vis_order)
{
  assert(m_composite_mode == VIS_ORDER_BLEND);
  Image image;
  const size_t image_index = m_images.size();
  m_images.push_back(image);
  bool gl_depth = false;
  m_images[image_index].Init(color_buffer,
                             depth_buffer,
                             width,
                             height,
                             gl_depth,
                             vis_order);
}

Image
Compositor::Composite()
{
  assert(m_images.size() != 0);

  if(m_composite_mode == Z_BUFFER_SURFACE_GL ||
     m_composite_mode == Z_BUFFER_SURFACE_WORLD )
  {
    CompositeZBufferSurface();
  }
  else if(m_composite_mode == Z_BUFFER_BLEND)
  {
    CompositeZBufferBlend();
  }
  else if(m_composite_mode == VIS_ORDER_BLEND)
  {
    CompositeVisOrder();
  }
  // TODO Make this a param/ref to avoid the copy?
  return m_images[0];
}

void
Compositor::Cleanup()
{

}

std::string
Compositor::GetLogString()
{
  std::string res = m_log_stream.str();
  m_log_stream.str("");
  return res;
}

void
Compositor::CompositeZBufferSurface()
{
  // nothing to do here in serial. Images were composited as
  // they were added to the compositor
#ifdef APCOMP_PARALLEL
  apcompdiy::mpi::communicator diy_comm;
  diy_comm = apcompdiy::mpi::communicator(MPI_Comm_f2c(mpi_comm()));

  assert(m_images.size() == 1);
  RadixKCompositor compositor;
  compositor.CompositeSurface(diy_comm, this->m_images[0]);
  m_log_stream<<compositor.GetTimingString();
#endif
}

void
Compositor::CompositeZBufferBlend()
{
  assert("this is not implemented yet" == "error");
}

void
Compositor::CompositeVisOrder()
{

#ifdef APCOMP_PARALLEL
  apcompdiy::mpi::communicator diy_comm;
  diy_comm = apcompdiy::mpi::communicator(MPI_Comm_f2c(mpi_comm()));

  assert(m_images.size() != 0);
  DirectSendCompositor compositor;
  compositor.CompositeVolume(diy_comm, this->m_images);
#else
  apcomp::ImageCompositor compositor;
  compositor.OrderedComposite(m_images);
#endif
}

} // namespace apcomp


