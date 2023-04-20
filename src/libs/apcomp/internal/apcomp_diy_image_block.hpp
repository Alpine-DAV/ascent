//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_DIY_IMAGE_BLOCK_HPP
#define APCOMP_DIY_IMAGE_BLOCK_HPP

#include <apcomp/apcomp_config.h>

#include <apcomp/image.hpp>
#include <apcomp/scalar_image.hpp>
#include <diy/master.hpp>

namespace apcomp
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
  const apcompdiy::Master &m_master;

  AddImageBlock(apcompdiy::Master &master, ImageType &image)
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
    apcompdiy::Master& master = const_cast<apcompdiy::Master&>(m_master);
    master.add(gid, block, linked);
  }
};

struct AddMultiImageBlock
{
  std::vector<Image> &m_images;
  Image              &m_output;
  const apcompdiy::Master  &m_master;

  AddMultiImageBlock(apcompdiy::Master &master,
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
    apcompdiy::Master& master = const_cast<apcompdiy::Master&>(m_master);
    int lid = master.add(gid, block, linked);
  }
};

} //namespace  apcomp

namespace apcompdiy {

template<>
struct Serialization<apcomp::ScalarImage>
{
  static void save(BinaryBuffer &bb, const apcomp::ScalarImage &image)
  {
    apcompdiy::save(bb, image.m_orig_bounds.m_min_x);
    apcompdiy::save(bb, image.m_orig_bounds.m_min_y);
    apcompdiy::save(bb, image.m_orig_bounds.m_max_x);
    apcompdiy::save(bb, image.m_orig_bounds.m_max_y);

    apcompdiy::save(bb, image.m_bounds.m_min_x);
    apcompdiy::save(bb, image.m_bounds.m_min_y);
    apcompdiy::save(bb, image.m_bounds.m_max_x);
    apcompdiy::save(bb, image.m_bounds.m_max_y);

    apcompdiy::save(bb, image.m_payloads);
    apcompdiy::save(bb, image.m_payload_bytes);
    apcompdiy::save(bb, image.m_depths);
    apcompdiy::save(bb, image.m_orig_rank);
  }

  static void load(BinaryBuffer &bb, apcomp::ScalarImage &image)
  {
    apcompdiy::load(bb, image.m_orig_bounds.m_min_x);
    apcompdiy::load(bb, image.m_orig_bounds.m_min_y);
    apcompdiy::load(bb, image.m_orig_bounds.m_max_x);
    apcompdiy::load(bb, image.m_orig_bounds.m_max_y);

    apcompdiy::load(bb, image.m_bounds.m_min_x);
    apcompdiy::load(bb, image.m_bounds.m_min_y);
    apcompdiy::load(bb, image.m_bounds.m_max_x);
    apcompdiy::load(bb, image.m_bounds.m_max_y);

    apcompdiy::load(bb, image.m_payloads);
    apcompdiy::load(bb, image.m_payload_bytes);
    apcompdiy::load(bb, image.m_depths);
    apcompdiy::load(bb, image.m_orig_rank);
  }
};

template<>
struct Serialization<apcomp::Image>
{
  static void save(BinaryBuffer &bb, const apcomp::Image &image)
  {
    apcompdiy::save(bb, image.m_orig_bounds.m_min_x);
    apcompdiy::save(bb, image.m_orig_bounds.m_min_y);
    apcompdiy::save(bb, image.m_orig_bounds.m_max_x);
    apcompdiy::save(bb, image.m_orig_bounds.m_max_y);

    apcompdiy::save(bb, image.m_bounds.m_min_x);
    apcompdiy::save(bb, image.m_bounds.m_min_y);
    apcompdiy::save(bb, image.m_bounds.m_max_x);
    apcompdiy::save(bb, image.m_bounds.m_max_y);

    apcompdiy::save(bb, image.m_pixels);
    apcompdiy::save(bb, image.m_depths);
    apcompdiy::save(bb, image.m_orig_rank);
    apcompdiy::save(bb, image.m_composite_order);
  }

  static void load(BinaryBuffer &bb, apcomp::Image &image)
  {
    apcompdiy::load(bb, image.m_orig_bounds.m_min_x);
    apcompdiy::load(bb, image.m_orig_bounds.m_min_y);
    apcompdiy::load(bb, image.m_orig_bounds.m_max_x);
    apcompdiy::load(bb, image.m_orig_bounds.m_max_y);

    apcompdiy::load(bb, image.m_bounds.m_min_x);
    apcompdiy::load(bb, image.m_bounds.m_min_y);
    apcompdiy::load(bb, image.m_bounds.m_max_x);
    apcompdiy::load(bb, image.m_bounds.m_max_y);

    apcompdiy::load(bb, image.m_pixels);
    apcompdiy::load(bb, image.m_depths);
    apcompdiy::load(bb, image.m_orig_rank);
    apcompdiy::load(bb, image.m_composite_order);
  }
};

} // namespace diy

#endif
