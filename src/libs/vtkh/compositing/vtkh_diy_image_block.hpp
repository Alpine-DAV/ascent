#ifndef VTKH_DIY_IMAGE_BLOCK_HPP
#define VTKH_DIY_IMAGE_BLOCK_HPP

#include <vtkh/compositing/Image.hpp>
#include <vtkh/compositing/PayloadImage.hpp>
#include <diy/master.hpp>

namespace vtkh
{

template<typename ImageType>
struct ImageBlock
{
  ImageType &m_image;
  ImageBlock(ImageType &image)
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

template<typename ImageType>
struct AddImageBlock
{
  ImageType             &m_image;
  const vtkhdiy::Master &m_master;

  AddImageBlock(vtkhdiy::Master &master, ImageType &image)
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
    ImageBlock<ImageType> *block = new ImageBlock<ImageType>(m_image);
    LinkType *linked = new LinkType(link);
    vtkhdiy::Master& master = const_cast<vtkhdiy::Master&>(m_master);
    master.add(gid, block, linked);
  }
};

struct AddMultiImageBlock
{
  std::vector<Image> &m_images;
  Image              &m_output;
  const vtkhdiy::Master  &m_master;

  AddMultiImageBlock(vtkhdiy::Master &master,
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
    vtkhdiy::Master& master = const_cast<vtkhdiy::Master&>(m_master);
    int lid = master.add(gid, block, linked);
  }
};

} //namespace  vtkh

namespace vtkhdiy {

template<>
struct Serialization<vtkh::PayloadImage>
{
  static void save(BinaryBuffer &bb, const vtkh::PayloadImage &image)
  {
    vtkhdiy::save(bb, image.m_orig_bounds.X.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.Y.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.Z.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.X.Max);
    vtkhdiy::save(bb, image.m_orig_bounds.Y.Max);
    vtkhdiy::save(bb, image.m_orig_bounds.Z.Max);

    vtkhdiy::save(bb, image.m_bounds.X.Min);
    vtkhdiy::save(bb, image.m_bounds.Y.Min);
    vtkhdiy::save(bb, image.m_bounds.Z.Min);
    vtkhdiy::save(bb, image.m_bounds.X.Max);
    vtkhdiy::save(bb, image.m_bounds.Y.Max);
    vtkhdiy::save(bb, image.m_bounds.Z.Max);

    vtkhdiy::save(bb, image.m_payloads);
    vtkhdiy::save(bb, image.m_payload_bytes);
    vtkhdiy::save(bb, image.m_depths);
    vtkhdiy::save(bb, image.m_orig_rank);
  }

  static void load(BinaryBuffer &bb, vtkh::PayloadImage &image)
  {
    vtkhdiy::load(bb, image.m_orig_bounds.X.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.Y.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.Z.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.X.Max);
    vtkhdiy::load(bb, image.m_orig_bounds.Y.Max);
    vtkhdiy::load(bb, image.m_orig_bounds.Z.Max);

    vtkhdiy::load(bb, image.m_bounds.X.Min);
    vtkhdiy::load(bb, image.m_bounds.Y.Min);
    vtkhdiy::load(bb, image.m_bounds.Z.Min);
    vtkhdiy::load(bb, image.m_bounds.X.Max);
    vtkhdiy::load(bb, image.m_bounds.Y.Max);
    vtkhdiy::load(bb, image.m_bounds.Z.Max);

    vtkhdiy::load(bb, image.m_payloads);
    vtkhdiy::load(bb, image.m_payload_bytes);
    vtkhdiy::load(bb, image.m_depths);
    vtkhdiy::load(bb, image.m_orig_rank);
  }
};

template<>
struct Serialization<vtkh::Image>
{
  static void save(BinaryBuffer &bb, const vtkh::Image &image)
  {
    vtkhdiy::save(bb, image.m_orig_bounds.X.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.Y.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.Z.Min);
    vtkhdiy::save(bb, image.m_orig_bounds.X.Max);
    vtkhdiy::save(bb, image.m_orig_bounds.Y.Max);
    vtkhdiy::save(bb, image.m_orig_bounds.Z.Max);

    vtkhdiy::save(bb, image.m_bounds.X.Min);
    vtkhdiy::save(bb, image.m_bounds.Y.Min);
    vtkhdiy::save(bb, image.m_bounds.Z.Min);
    vtkhdiy::save(bb, image.m_bounds.X.Max);
    vtkhdiy::save(bb, image.m_bounds.Y.Max);
    vtkhdiy::save(bb, image.m_bounds.Z.Max);

    vtkhdiy::save(bb, image.m_pixels);
    vtkhdiy::save(bb, image.m_depths);
    vtkhdiy::save(bb, image.m_orig_rank);
    vtkhdiy::save(bb, image.m_composite_order);
  }

  static void load(BinaryBuffer &bb, vtkh::Image &image)
  {
    vtkhdiy::load(bb, image.m_orig_bounds.X.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.Y.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.Z.Min);
    vtkhdiy::load(bb, image.m_orig_bounds.X.Max);
    vtkhdiy::load(bb, image.m_orig_bounds.Y.Max);
    vtkhdiy::load(bb, image.m_orig_bounds.Z.Max);

    vtkhdiy::load(bb, image.m_bounds.X.Min);
    vtkhdiy::load(bb, image.m_bounds.Y.Min);
    vtkhdiy::load(bb, image.m_bounds.Z.Min);
    vtkhdiy::load(bb, image.m_bounds.X.Max);
    vtkhdiy::load(bb, image.m_bounds.Y.Max);
    vtkhdiy::load(bb, image.m_bounds.Z.Max);

    vtkhdiy::load(bb, image.m_pixels);
    vtkhdiy::load(bb, image.m_depths);
    vtkhdiy::load(bb, image.m_orig_rank);
    vtkhdiy::load(bb, image.m_composite_order);
  }
};

} // namespace diy

#endif
