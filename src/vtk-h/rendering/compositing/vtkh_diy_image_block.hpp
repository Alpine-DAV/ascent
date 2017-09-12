#ifndef VTKH_DIY_IMAGE_BLOCK_HPP
#define VTKH_DIY_IMAGE_BLOCK_HPP

#include <rendering/vtkh_image.hpp>
#include <diy/master.hpp>

namespace vtkh 
{

struct ImageBlock
{
  Image &m_image;
  ImageBlock(Image &image)
    : m_image(image)
  {
  }
};

struct MultiImageBlock
{
  std::vector<Image> &m_images;
  Image              &m_output;
  MultiImageBlock(std::vector<Image> &images,
                  Image &output)
    : m_images(images),
      m_output(output)
  {}
};

struct AddImageBlock
{
  Image             &m_image;
  const diy::Master &m_master;

  AddImageBlock(diy::Master &master, Image &image)
    : m_image(image),
      m_master(master)
  {
  }
  template<typename BoundsType, typename LinkType>                 
  void operator()(int gid,
                  const BoundsType &,  // local_bounds
                  const BoundsType &,  // local_with_ghost_bounds
                  const BoundsType &,  // domain_bounds
                  const LinkType &link) const
  {
    ImageBlock *block = new ImageBlock(m_image);
    LinkType *linked = new LinkType(link);
    diy::Master& master = const_cast<diy::Master&>(m_master);
    master.add(gid, block, linked);
  }
}; 

struct AddMultiImageBlock
{
  std::vector<Image> &m_images;
  Image              &m_output;
  const diy::Master  &m_master;

  AddMultiImageBlock(diy::Master &master, 
                     std::vector<Image> &images,
                     Image &output)
    : m_master(master), 
      m_images(images),
      m_output(output)
  {}
  template<typename BoundsType, typename LinkType>                 
  void operator()(int gid,
                  const BoundsType &,  // local_bounds
                  const BoundsType &,  // local_with_ghost_bounds
                  const BoundsType &,  // domain_bounds
                  const LinkType &link) const
  {
    MultiImageBlock *block = new MultiImageBlock(m_images, m_output);
    LinkType *linked = new LinkType(link);
    diy::Master& master = const_cast<diy::Master&>(m_master);
    int lid = master.add(gid, block, linked);
  }
}; 

} //namespace  vtkh

namespace diy {

template<>
struct Serialization<vtkh::Image>
{
  static void save(BinaryBuffer &bb, const vtkh::Image &image)
  {
    diy::save(bb, image.m_orig_bounds.X.Min);
    diy::save(bb, image.m_orig_bounds.Y.Min);
    diy::save(bb, image.m_orig_bounds.Z.Min);
    diy::save(bb, image.m_orig_bounds.X.Max);
    diy::save(bb, image.m_orig_bounds.Y.Max);
    diy::save(bb, image.m_orig_bounds.Z.Max);

    diy::save(bb, image.m_bounds.X.Min);
    diy::save(bb, image.m_bounds.Y.Min);
    diy::save(bb, image.m_bounds.Z.Min);
    diy::save(bb, image.m_bounds.X.Max);
    diy::save(bb, image.m_bounds.Y.Max);
    diy::save(bb, image.m_bounds.Z.Max);

    diy::save(bb, image.m_pixels);
    diy::save(bb, image.m_depths);
    diy::save(bb, image.m_orig_rank);
    diy::save(bb, image.m_z_buffer_mode);
    diy::save(bb, image.m_composite_order);
  }

  static void load(BinaryBuffer &bb, vtkh::Image &image)
  {
    diy::load(bb, image.m_orig_bounds.X.Min);
    diy::load(bb, image.m_orig_bounds.Y.Min);
    diy::load(bb, image.m_orig_bounds.Z.Min);
    diy::load(bb, image.m_orig_bounds.X.Max);
    diy::load(bb, image.m_orig_bounds.Y.Max);
    diy::load(bb, image.m_orig_bounds.Z.Max);

    diy::load(bb, image.m_bounds.X.Min);
    diy::load(bb, image.m_bounds.Y.Min);
    diy::load(bb, image.m_bounds.Z.Min);
    diy::load(bb, image.m_bounds.X.Max);
    diy::load(bb, image.m_bounds.Y.Max);
    diy::load(bb, image.m_bounds.Z.Max);

    diy::load(bb, image.m_pixels);
    diy::load(bb, image.m_depths);
    diy::load(bb, image.m_orig_rank);
    diy::load(bb, image.m_z_buffer_mode);
    diy::load(bb, image.m_composite_order);
  }
};

} // namespace diy

#endif
