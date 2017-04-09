#ifndef ALPINE_DIY_IMAGE_HPP
#define ALPINE_DIY_IMAGE_HPP

#include <diy/master.hpp>
#include <alpine_png_encoder.hpp>
#include <alpine_config.h>

// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

struct Image
{
    // The image bounds are indicated by a grid starting at
    // 1-width and 1-height. Actual width would be calculated 
    // m_bounds.max[0] - m_bounds.min[0] + 1
    // 1024 - 1 + 1 = 1024
    diy::DiscreteBounds          m_orig_bounds; 
    diy::DiscreteBounds          m_bounds; 
    std::vector<unsigned char>   m_pixels;
    std::vector<float>           m_depths; 
    int                          m_orig_rank;
    bool                         m_z_buffer_mode;
    Image()
    {}

    Image(const diy::DiscreteBounds &bounds, bool z_buffer_mode = true)
      : m_orig_bounds(bounds),
        m_bounds(bounds),
        m_orig_rank(-1),
        m_z_buffer_mode(z_buffer_mode)

    {
        const int dx  = bounds.max[0] - bounds.min[0] + 1;
        const int dy  = bounds.max[1] - bounds.min[1] + 1;
        m_pixels.resize(dx * dy * 4);
        if(m_z_buffer_mode)
        {
          m_depths.resize(dx * dy);
        }
    }

    void Init(const float *color_buffer,
              const float *depth_buffer,
              int width,
              int height)
    {
      m_bounds.min[0] = 1;
      m_bounds.min[1] = 1;
      m_bounds.max[0] = width;
      m_bounds.max[1] = height;
      m_orig_bounds = m_bounds; 
      m_z_buffer_mode = depth_buffer != NULL; 
      const int size = width * height;
      m_pixels.resize(size * 4);
      if(m_z_buffer_mode)
      {
        m_depths.resize(size);
      }
#ifdef ALPINE_USE_OPENMP
      #pragma omp parallel for 
#endif
      for(int i = 0; i < size; ++i)
      {
        const int offset = i * 4;
        m_pixels[offset + 0] = static_cast<unsigned char>(color_buffer[offset + 0] * 255.f);
        m_pixels[offset + 1] = static_cast<unsigned char>(color_buffer[offset + 1] * 255.f);
        m_pixels[offset + 2] = static_cast<unsigned char>(color_buffer[offset + 2] * 255.f);
        m_pixels[offset + 3] = static_cast<unsigned char>(color_buffer[offset + 3] * 255.f);
        if(m_z_buffer_mode)
        {
          float depth = depth_buffer[i];
          //make sure we can do a single comparison on depth
          depth = depth < 0 ? 2.f : depth;
          m_depths[i] =  depth;
        }
      }
      /*
      std::copy(depth_buffer,
                depth_buffer + size,
                &m_depths[0]);
      */
    }

    void Init(const unsigned char *color_buffer,
              const float *depth_buffer,
              int width,
              int height)
    {
      m_bounds.min[0] = 1;
      m_bounds.min[1] = 1;
      m_bounds.max[0] = width;
      m_bounds.max[1] = height;
      m_orig_bounds = m_bounds; 

      const int size = width * height;
      m_pixels.resize(size * 4);
      m_z_buffer_mode = depth_buffer != NULL;
      if(m_z_buffer_mode)
      {
        m_depths.resize(size);
      }

      std::copy(color_buffer,
                color_buffer + size * 4,
                &m_pixels[0]);

      if(m_z_buffer_mode)
      {
#ifdef ALPINE_USE_OPENMP
        #pragma omp parallel for 
#endif
        for(int i = 0; i < size; ++i)
        {
          float depth = depth_buffer[i];
          //make sure we can do a single comparison on depth
          depth = depth < 0 ? 2.f : depth;
          m_depths[i] =  depth;
        } // for
      } // if depth
    }

    void Composite(const Image &image)
    {
      if(m_z_buffer_mode)
      {
        ZComposite(image);
      }
      else
      {
        Blend(image);
      }
    }

    void ZComposite(const Image &image)
    {
      assert(m_depths.size() == m_pixels.size());
      assert(m_bounds.min[0] == image.m_bounds.min[0]); 
      assert(m_bounds.min[1] == image.m_bounds.min[1]); 
      assert(m_bounds.max[0] == image.m_bounds.max[0]); 
      assert(m_bounds.max[1] == image.m_bounds.max[1]); 

      const int size = static_cast<int>(m_depths.size()); 
  
#ifdef ALPINE_USE_OPENMP
      #pragma omp parallel for 
#endif
      for(int i = 0; i < size; ++i)
      {
        const float depth = image.m_depths[i];
        if(depth > 1.f  || m_depths[i] < depth)
        {
          continue;
        }
        const int offset = i * 4;
        m_depths[i] = depth;
        m_pixels[offset + 0] = image.m_pixels[offset + 0];
        m_pixels[offset + 1] = image.m_pixels[offset + 1];
        m_pixels[offset + 2] = image.m_pixels[offset + 2];
        m_pixels[offset + 3] = image.m_pixels[offset + 3];
      }
    }
    void PrintColor(int i, float alpha) const
    {
      std::cout<<"["<<(int)m_pixels[i*4+0]<<","
                    <<(int)m_pixels[i*4+1]<<","
                    <<(int)m_pixels[i*4+2]<<","<<alpha<<"]\n";
    } 

    void Blend(const Image &image)
    {
      assert(m_bounds.min[0] == image.m_bounds.min[0]); 
      assert(m_bounds.min[1] == image.m_bounds.min[1]); 
      assert(m_bounds.max[0] == image.m_bounds.max[0]); 
      assert(m_bounds.max[1] == image.m_bounds.max[1]); 

      const int size = static_cast<int>(m_pixels.size() / 4); 
  
#ifdef ALPINE_USE_OPENMP
      #pragma omp parallel for 
#endif
      for(int i = 0; i < size; ++i)
      {
        const int offset = i * 4;
        //float alpha = static_cast<float>(m_pixels[offset + 3]) / 255.f;
        unsigned int alpha = m_pixels[offset + 3];// / 255.f;
        //const float opacity = (1.f - alpha) * alpha2;
        const unsigned int opacity = 255 - alpha;//(1.f - alpha) * alpha2;
        //m_pixels[offset + 0] += static_cast<unsigned char>(opacity * static_cast<float>(image.m_pixels[offset + 0])); 
        //m_pixels[offset + 1] += static_cast<unsigned char>(opacity * static_cast<float>(image.m_pixels[offset + 1])); 
        //m_pixels[offset + 2] += static_cast<unsigned char>(opacity * static_cast<float>(image.m_pixels[offset + 2])); 
        m_pixels[offset + 0] += static_cast<unsigned char>(opacity * image.m_pixels[offset + 0] / 255); 
        m_pixels[offset + 1] += static_cast<unsigned char>(opacity * image.m_pixels[offset + 1] / 255); 
        m_pixels[offset + 2] += static_cast<unsigned char>(opacity * image.m_pixels[offset + 2] / 255); 
        m_pixels[offset + 3] += static_cast<unsigned char>(opacity * image.m_pixels[offset + 3] / 255); 
      }
    }
    
    void CompositeBackground(const float *color)
    {

      const int size = static_cast<int>(m_pixels.size() / 4); 
      std::cout<<"BG "<<color[0]<<" "<<color[1]<<" "<<color[2]<<" "<<color[3]<<"\n"; 
      unsigned char bg_color[4];
      for(int i = 0; i < 4; ++i)
      {
        bg_color[i] = static_cast<unsigned char>(color[i] * 255.f);
      }

#ifdef ALPINE_USE_OPENMP
      #pragma omp parallel for 
#endif
      for(int i = 0; i < size; ++i)
      {
        const int offset = i * 4;
        unsigned int alpha = static_cast<unsigned int>(m_pixels[offset + 3]);
        const float opacity = (255 - alpha);
        m_pixels[offset + 0] += static_cast<unsigned char>(opacity * bg_color[0] / 255); 
        m_pixels[offset + 1] += static_cast<unsigned char>(opacity * bg_color[1] / 255); 
        m_pixels[offset + 2] += static_cast<unsigned char>(opacity * bg_color[2] / 255); 
        m_pixels[offset + 3] += static_cast<unsigned char>(opacity * bg_color[3] / 255); 
      }
    }
    //
    // Fill this image with a sub-region of another image 
    //
    void SubsetFrom(const Image &image,
                    const diy::DiscreteBounds &sub_region)
    {
      m_orig_bounds = image.m_orig_bounds;
      m_bounds = sub_region;
      m_orig_rank = image.m_orig_rank;
      m_z_buffer_mode = image.m_z_buffer_mode;

      assert(sub_region.min[0] >= image.m_bounds.min[0]);
      assert(sub_region.min[1] >= image.m_bounds.min[1]);
      assert(sub_region.max[0] <= image.m_bounds.max[0]);
      assert(sub_region.max[1] <= image.m_bounds.max[1]);

      const int s_dx  = m_bounds.max[0] - m_bounds.min[0] + 1;
      const int s_dy  = m_bounds.max[1] - m_bounds.min[1] + 1;

      const int dx  = image.m_bounds.max[0] - image.m_bounds.min[0] + 1;
      const int dy  = image.m_bounds.max[1] - image.m_bounds.min[1] + 1;
      
      const int start_x = m_bounds.min[0] - image.m_bounds.min[0];
      const int start_y = m_bounds.min[1] - image.m_bounds.min[1];
      const int end_y = start_y + s_dy;

      m_pixels.resize(s_dx * s_dy * 4);

      if(m_z_buffer_mode)
      {
        m_depths.resize(s_dx * s_dy);
      }
      
      
#ifdef ALPINE_USE_OPENMP
        #pragma omp parallel for 
#endif
      for(int y = start_y; y < end_y; ++y)
      {
        const int copy_to = (y - start_y) * s_dx;
        const int copy_from = y * dx + start_x;

        std::copy(&image.m_pixels[copy_from * 4],
                  &image.m_pixels[copy_from * 4] + s_dx * 4,
                  &m_pixels[copy_to * 4]);
        if(m_z_buffer_mode)
        {
          std::copy(&image.m_depths[copy_from],
                    &image.m_depths[copy_from] + s_dx,
                    &m_depths[copy_to]);
        }
      }
      
    }
    
    void Color(int color)
    {
      unsigned char c[4];   
      c[3] = 255;

      c[0] = 0;
      c[1] = 0;
      c[2] = 0;
      int index = color % 3; 
      c[index] = 255 - color * 11;;
      for(int i = 0; i < m_pixels.size(); ++i)
      {
        float d = m_depths[i / 4]; 
        if(d >0 && d < 1)
        {
         m_pixels[i] = c[i%4]; 
        }
        else
        {
         m_pixels[i] = 155; 
        }
      }

    }
    //
    // Fills the passed in image with the contents of this image
    //
    void SubsetTo(Image &image) const
    {
     
      assert(m_bounds.min[0] >= image.m_bounds.min[0]);
      assert(m_bounds.min[1] >= image.m_bounds.min[1]);
      assert(m_bounds.max[0] <= image.m_bounds.max[0]);
      assert(m_bounds.max[1] <= image.m_bounds.max[1]);

      const int s_dx  = m_bounds.max[0] - m_bounds.min[0] + 1;
      const int s_dy  = m_bounds.max[1] - m_bounds.min[1] + 1;

      const int dx  = image.m_bounds.max[0] - image.m_bounds.min[0] + 1;
      const int dy  = image.m_bounds.max[1] - image.m_bounds.min[1] + 1;
      
      const int start_x = m_bounds.min[0] - image.m_bounds.min[0];
      const int start_y = m_bounds.min[1] - image.m_bounds.min[1];
    
#ifdef ALPINE_USE_OPENMP
        #pragma omp parallel for 
#endif
      for(int y = 0; y < s_dy; ++y)
      {
        const int copy_to = (y + start_y) * dx + start_x;
        const int copy_from = y * s_dx;
        
        std::copy(&m_pixels[copy_from * 4],
                  &m_pixels[copy_from * 4] + s_dx * 4,
                  &image.m_pixels[copy_to * 4]);

        if(m_z_buffer_mode)
        {
          std::copy(&m_depths[copy_from],
                    &m_depths[copy_from] + s_dx,
                    &image.m_depths[copy_to]);
        }
      }
    }

    void Swap(Image &other)
    {
      diy::DiscreteBounds orig   = m_orig_bounds;
      diy::DiscreteBounds bounds = m_bounds;

      m_orig_bounds = other.m_orig_bounds;
      m_bounds      = other.m_bounds;

      other.m_orig_bounds = orig;
      other.m_bounds      = bounds;

      m_pixels.swap(other.m_pixels);
      m_depths.swap(other.m_depths);
    }
    
    void Clear()
    { 
      diy::DiscreteBounds empty;
      m_orig_bounds = empty;
      m_bounds = empty;
      m_pixels.clear();
      m_depths.clear();
    }

    std::string ToString() const
    {
      std::stringstream ss;
      fmt::print(ss, "Total size pixels {} tile dims: [{},{}] - [{},{}]\n",
                 (int) m_pixels.size() / 4, 
                 m_bounds.min[0],
                 m_bounds.min[1],
                 m_bounds.max[0],
                 m_bounds.max[1]);
      return ss.str();
    }

    void Save(std::string name)
    {
      PNGEncoder encoder;
      encoder.Encode(&m_pixels[0],
                     m_bounds.max[0] - m_bounds.min[0] + 1,
                     m_bounds.max[1] - m_bounds.min[1] + 1);
      encoder.Save(name);
    }
};

struct ImageBlock
{
  Image &m_image;
  ImageBlock(Image &image)
    : m_image(image)
  {}
};

struct AddImageBlock
{
  Image         &m_image;
  const diy::Master  &m_master;

  AddImageBlock(diy::Master &master, Image &image)
    : m_master(master), m_image(image)
  {}
  template<typename BoundsType, typename LinkType>                 
  void operator()(int gid,
                  const BoundsType &local_bounds,
                  const BoundsType &local_with_ghost_bounds,
                  const BoundsType &domain_bounds,
                  const LinkType &link) const
  {
    ImageBlock *block = new ImageBlock(m_image);
    LinkType *linked = new LinkType(link);
    diy::Master& master = const_cast<diy::Master&>(m_master);
    int lid = master.add(gid, block, linked);
  }
}; 

} //namespace alpine

namespace diy {

template<>
struct Serialization<alpine::Image>
{
  static void save(BinaryBuffer &bb, const alpine::Image &image)
  {
    diy::save(bb, image.m_orig_bounds);
    diy::save(bb, image.m_bounds);
    diy::save(bb, image.m_pixels);
    diy::save(bb, image.m_depths);
    diy::save(bb, image.m_orig_rank);
    diy::save(bb, image.m_z_buffer_mode);
  }

  static void load(BinaryBuffer &bb, alpine::Image &image)
  {
    diy::load(bb, image.m_orig_bounds);
    diy::load(bb, image.m_bounds);
    diy::load(bb, image.m_pixels);
    diy::load(bb, image.m_depths);
    diy::load(bb, image.m_orig_rank);
    diy::load(bb, image.m_z_buffer_mode);
  }
};

} // namespace diy

#endif
