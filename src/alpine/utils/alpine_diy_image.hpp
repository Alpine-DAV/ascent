#ifndef ALPINE_DIY_IMAGE_HPP
#define ALPINE_DIY_IMAGE_HPP

#include <diy/master.hpp>
#include <alpine_png_encoder.hpp>
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

struct Image
{
    diy::DiscreteBounds          m_orig_bounds; 
    diy::DiscreteBounds          m_bounds; 
    std::vector<unsigned char>   m_pixels;
    std::vector<float>           m_depths; 

    Image()
    {}

    Image(const diy::DiscreteBounds &bounds)
      : m_orig_bounds(bounds),
        m_bounds(bounds) 
    {
        const int dx  = bounds.max[0] - bounds.min[0];
        const int dy  = bounds.max[1] - bounds.min[1];
        m_pixels.resize(dx * dy * 4);
        m_depths.resize(dx * dy);
    }

    void Init(const float *color_buffer,
              const float *depth_buffer,
              int width,
              int height)
    {
      m_bounds.min[0] = 0;
      m_bounds.min[1] = 0;
      m_bounds.max[0] = width;
      m_bounds.max[1] = height;
      m_orig_bounds = m_bounds; 
    
      const int size = width * height;
      m_pixels.resize(size * 4);
      m_depths.resize(size);
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
        float depth = depth_buffer[i];
        //make sure we can do a single comparison on depth
        //depth = depth < 0 ? 2.f : depth;
        m_depths[i] =  depth;
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
      m_bounds.min[0] = 0;
      m_bounds.min[1] = 0;
      m_bounds.max[0] = width;
      m_bounds.max[1] = height;
      m_orig_bounds = m_bounds; 

      const int size = width * height;
      m_pixels.resize(size * 4);
      m_depths.resize(size);

      std::copy(color_buffer,
                color_buffer + size * 4,
                &m_pixels[0]);
#ifdef ALPINE_USE_OPENMP
      #pragma omp parallel for 
#endif
      for(int i = 0; i < size; ++i)
      {
        float depth = depth_buffer[i];
        //make sure we can do a single comparison on depth
        //depth = depth < 0 ? 2.f : depth;
        m_depths[i] =  depth;
      }
    }

    void Composite(const Image &image)
    {
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
        if(m_depths[i] < image.m_depths[i])
        {
          continue;
        }
        const int offset = i * 4;
        m_depths[i] = image.m_depths[i];
        m_pixels[offset + 0] = image.m_pixels[offset + 0];
        m_pixels[offset + 1] = image.m_pixels[offset + 1];
        m_pixels[offset + 2] = image.m_pixels[offset + 2];
        m_pixels[offset + 3] = image.m_pixels[offset + 3];
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

      assert(sub_region.min[0] >= image.m_bounds.min[0]);
      assert(sub_region.min[1] >= image.m_bounds.min[1]);
      assert(sub_region.max[0] <= image.m_bounds.max[0]);
      assert(sub_region.max[1] <= image.m_bounds.max[1]);

      const int s_dx  = m_bounds.max[0] - m_bounds.min[0];
      const int s_dy  = m_bounds.max[1] - m_bounds.min[1];

      const int dx  = image.m_bounds.max[0] - image.m_bounds.min[0];
      const int dy  = image.m_bounds.max[1] - image.m_bounds.min[1];
      
      const int start_x = m_bounds.min[0] - image.m_bounds.min[0];
      const int start_y = m_bounds.min[1] - image.m_bounds.min[1];
      const int end_y = start_y + s_dy;
      m_pixels.resize(s_dx * s_dy * 4);
      m_depths.resize(s_dx * s_dy);
      
      
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

        std::copy(&image.m_depths[copy_from],
                  &image.m_depths[copy_from] + s_dx,
                  &m_depths[copy_to]);
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

      const int s_dx  = m_bounds.max[0] - m_bounds.min[0];
      const int s_dy  = m_bounds.max[1] - m_bounds.min[1];

      const int dx  = image.m_bounds.max[0] - image.m_bounds.min[0];
      const int dy  = image.m_bounds.max[1] - image.m_bounds.min[1];
      
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
        std::copy(&m_depths[copy_from],
                  &m_depths[copy_from] + s_dx,
                  &image.m_depths[copy_to]);
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
      fmt::print(ss, "Total size pixels {} tile dims: [{},{}] - [{},{}] {}\n",
                 (int) m_depths.size(), 
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
                     m_bounds.max[0] - m_bounds.min[0],
                     m_bounds.max[1] - m_bounds.min[0]);
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
  }

  static void load(BinaryBuffer &bb, alpine::Image &image)
  {
    diy::load(bb, image.m_orig_bounds);
    diy::load(bb, image.m_bounds);
    diy::load(bb, image.m_pixels);
    diy::load(bb, image.m_depths);
  }
};

} // namespace diy

#endif
