#include "vtkh_compositor.hpp"
#include <rendering/vtkh_image_compositor.hpp>

#include <assert.h>
#include <algorithm>

namespace vtkh 
{

Compositor::Compositor() 
  : m_composite_mode(Z_BUFFER_SURFACE),
    m_background_color{1.f, 1.f, 1.f, 1.f}
{ 

}

Compositor::~Compositor() 
{

}

void 
Compositor::SetBackgroundColor(float background_color[4])
{
  m_background_color[0] = background_color[0];
  m_background_color[1] = background_color[1];
  m_background_color[2] = background_color[2];
  m_background_color[3] = background_color[3];
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
static int count = 0;

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
    vtkh::ImageCompositor compositor;
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

    vtkh::ImageCompositor compositor;
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
                     const int            width,
                     const int            height,
                     const int            vis_order)
{
  assert(m_composite_mode == VIS_ORDER_BLEND);
  Image image;
  const size_t image_index = m_images.size();
  m_images.push_back(image);
  m_images[image_index].Init(color_buffer,
                             NULL,
                             width,
                             height,
                             vis_order);
}

void 
Compositor::AddImage(const float *color_buffer,
                     const int    width,
                     const int    height,
                     const int    vis_order)
{
  assert(m_composite_mode == VIS_ORDER_BLEND);
  Image image;
  const size_t image_index = m_images.size();
  m_images.push_back(image);

  m_images[image_index].Init(color_buffer,
                             NULL,
                             width,
                             height,
                             vis_order);
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
  vtkh::ImageCompositor compositor;
  compositor.OrderedComposite(m_images);
}

} // namespace vtkh


