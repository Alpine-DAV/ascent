#include "vtkh_compositor.hpp"

#include <assert.h>
#include <algorithm>

namespace vtkh 
{
namespace detail
{
  struct LocalVisOrder
  {
    int m_image_pos;   // the position of the image in an array
    int m_image_order; // composite order

    bool operator<(const LocalVisOrder &other) const
    {
      return m_image_order < other.m_image_order;
    }
  };

  
} // namespace detail

Compositor::Compositor() 
  : m_composite_mode(Z_BUFFER_SURFACE)
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
  if(m_images.size() == 0)
  {
    m_images.push_back(image);
    m_images[0].Init(color_buffer,
                     depth_buffer,
                     width,
                     height);
  }
  else if(m_composite_mode == Z_BUFFER_SURFACE)
  {
    //
    // Do local composite and keep a single image
    //
    image.Init(color_buffer,
               depth_buffer,
               width,
               height);

    m_images[0].Composite(image);
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
  if(m_images.size() == 0)
  {
    m_images.push_back(image);
    m_images[0].Init(color_buffer,
                     depth_buffer,
                     width,
                     height);
  }
  else if(m_composite_mode == Z_BUFFER_SURFACE)
  {
    //
    // Do local composite and keep a single image
    //
    image.Init(color_buffer,
               depth_buffer,
               width,
               height);

    m_images[0].Composite(image);
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
                     const int            width,
                     const int            height,
                     const int            vis_order)
{
  assert(m_composite_mode == VIS_ORDER_BLEND);
  Image image;
  const size_t image_index = m_images.size();
  m_images.push_back(image);
  m_local_vis_order.push_back(vis_order);
  m_images[image_index].Init(color_buffer,
                             NULL,
                             width,
                             height);
}

Image 
Compositor::Composite()
{
  assert(m_images.size() != 0);

  if(m_composite_mode == Z_BUFFER_SURFACE)
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
  // Make this a param to avoid the copy?
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
  // nothing to do here. Images were composited as 
  // they were added to the compositor
}

void 
Compositor::CompositeZBufferBlend()
{
  assert("this is not implemented yet" == "error");  
}

void 
Compositor::CompositeVisOrder()
{
  const int num_images = static_cast<int>(m_images.size()); 
  std::vector<detail::LocalVisOrder> vis_order;
  vis_order.resize(num_images);
  for(int i = 0; i < num_images; ++i)
  {
    vis_order[i].m_image_pos = i;
    vis_order[i].m_image_order = m_local_vis_order[i];
  }

  std::sort(vis_order.begin(), vis_order.end());
  
  const int first = vis_order[0].m_image_pos;
  for(int i = 1; i < num_images; ++i)
  {
    const int next = vis_order[i].m_image_pos;
    m_images[first].Composite(m_images[next]);
  }
  
  if(first != 0)
  {
    m_images[0].Swap(m_images[first]);
  }

}

} // namespace vtkh


