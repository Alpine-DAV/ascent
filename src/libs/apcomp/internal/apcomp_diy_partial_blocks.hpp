//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_IMAGE_BLOCKS_h
#define APCOMP_IMAGE_BLOCKS_h

#include <apcomp/apcomp_config.h>

#include <diy/master.hpp>

#include <apcomp/absorption_partial.hpp>
#include <apcomp/emission_partial.hpp>
#include <apcomp/volume_partial.hpp>

namespace apcomp {

//---------------Volume Block Structure---------------------
template<typename FloatType>
struct VolumeBlock
{
  typedef apcompdiy::DiscreteBounds            Bounds;
  typedef VolumePartial<FloatType>       PartialType;
  std::vector<VolumePartial<FloatType>> &m_partials;
  VolumeBlock(std::vector<VolumePartial<FloatType>> &partials)
    : m_partials(partials)
  {}
};


//--------------------------------------Absorption Block Structure------------------------------
template<typename FloatType>
struct AbsorptionBlock
{
  typedef apcompdiy::DiscreteBounds Bounds;
  typedef AbsorptionPartial<FloatType> PartialType;
  std::vector<AbsorptionPartial<FloatType>>   &m_partials;

  AbsorptionBlock(std::vector<AbsorptionPartial<FloatType>> &partials)
    : m_partials(partials)
  {
  }
};

//--------------------------------------Emission Block Structure------------------------------
template<typename FloatType>
struct EmissionBlock
{
  typedef apcompdiy::DiscreteBounds Bounds;
  typedef EmissionPartial<FloatType> PartialType;
  std::vector<EmissionPartial<FloatType>>   &m_partials;

  EmissionBlock(std::vector<EmissionPartial<FloatType>> &partials)
    : m_partials(partials)
  {}
};

//--------------------------------------Add Block Template-----------------------------------
template<typename BlockType>
struct AddBlock
{
  typedef typename BlockType::PartialType PartialType;
  typedef BlockType                       Block;
  std::vector<PartialType> &m_partials;
  const apcompdiy::Master &m_master;

  AddBlock(apcompdiy::Master &master,std::vector<PartialType> &partials)
    : m_master(master), m_partials(partials)
  {
  }
  template<typename BoundsType, typename LinkType>
  void operator()(int gid,
                  const BoundsType &local_bounds,
                  const BoundsType &local_with_ghost_bounds,
                  const BoundsType &domain_bounds,
                  const LinkType &link) const
  {
    (void) local_bounds;
    (void) domain_bounds;
    (void) local_with_ghost_bounds;
    Block *block = new Block(m_partials);
    LinkType *rg_link = new LinkType(link);
    apcompdiy::Master& master = const_cast<apcompdiy::Master&>(m_master);
    int lid = master.add(gid, block, rg_link);
    (void) lid;
  }
};

} //namespace apcomp

//-------------------------------Serialization Specializations--------------------------------
namespace apcompdiy {

template<>
struct Serialization<apcomp::AbsorptionPartial<double>>
{

  static void save(BinaryBuffer& bb, const apcomp::AbsorptionPartial<double> &partial)
  {
    apcompdiy::save(bb, partial.m_bins);
    apcompdiy::save(bb, partial.m_pixel_id);
    apcompdiy::save(bb, partial.m_depth);
  }

  static void load(BinaryBuffer& bb, apcomp::AbsorptionPartial<double> &partial)
  {
    apcompdiy::load(bb, partial.m_bins);
    apcompdiy::load(bb, partial.m_pixel_id);
    apcompdiy::load(bb, partial.m_depth);
  }
};

template<>
struct Serialization<apcomp::AbsorptionPartial<float>>
{

  static void save(BinaryBuffer& bb, const apcomp::AbsorptionPartial<float> &partial)
  {
    apcompdiy::save(bb, partial.m_bins);
    apcompdiy::save(bb, partial.m_pixel_id);
    apcompdiy::save(bb, partial.m_depth);
  }

  static void load(BinaryBuffer& bb, apcomp::AbsorptionPartial<float> &partial)
  {
    apcompdiy::load(bb, partial.m_bins);
    apcompdiy::load(bb, partial.m_pixel_id);
    apcompdiy::load(bb, partial.m_depth);
  }
};

template<>
struct Serialization<apcomp::EmissionPartial<double>>
{

  static void save(BinaryBuffer& bb, const apcomp::EmissionPartial<double> &partial)
  {
    apcompdiy::save(bb, partial.m_bins);
    apcompdiy::save(bb, partial.m_emission_bins);
    apcompdiy::save(bb, partial.m_pixel_id);
    apcompdiy::save(bb, partial.m_depth);
  }

  static void load(BinaryBuffer& bb, apcomp::EmissionPartial<double> &partial)
  {
    apcompdiy::load(bb, partial.m_bins);
    apcompdiy::load(bb, partial.m_emission_bins);
    apcompdiy::load(bb, partial.m_pixel_id);
    apcompdiy::load(bb, partial.m_depth);
  }
};

template<>
struct Serialization<apcomp::EmissionPartial<float>>
{

  static void save(BinaryBuffer& bb, const apcomp::EmissionPartial<float> &partial)
  {
    apcompdiy::save(bb, partial.m_bins);
    apcompdiy::save(bb, partial.m_emission_bins);
    apcompdiy::save(bb, partial.m_pixel_id);
    apcompdiy::save(bb, partial.m_depth);
  }

  static void load(BinaryBuffer& bb, apcomp::EmissionPartial<float> &partial)
  {
    apcompdiy::load(bb, partial.m_bins);
    apcompdiy::load(bb, partial.m_emission_bins);
    apcompdiy::load(bb, partial.m_pixel_id);
    apcompdiy::load(bb, partial.m_depth);
  }
};

} // namespace diy

#endif
