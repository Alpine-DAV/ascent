// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#ifndef DRAY_CLIP_HPP
#define DRAY_CLIP_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class Clip
{
public:
  Clip();
  virtual ~Clip();

  void SetBoxClip(const AABB<3> &bounds);
  void SetSphereClip(const Float center[3], const Float radius);
  void SetPlaneClip(const Float origin[3], const Float normal[3]);

  void Set2PlaneClip(const Float origin1[3],
                     const Float normal1[3],
                     const Float origin2[3],
                     const Float normal2[3]);

  void Set3PlaneClip(const Float origin1[3],
                     const Float normal1[3],
                     const Float origin2[3],
                     const Float normal2[3],
                     const Float origin3[3],
                     const Float normal3[3]);

  void SetInvertClip(bool invert);
  void SetMultiPlane(bool value);

  /**
   * @brief Clips the input dataset.
   * @param data_set The dataset.
   * @return A dataset containing the clipped cells.
   *
   */
  Collection execute(Collection &collection);

protected:
  struct InternalsType;
  std::shared_ptr<InternalsType> m_internals;
  bool m_invert;
  bool m_do_multi_plane; // cut planes one at a time
};

};//namespace dray

#endif//DRAY_CLIP_HPP
