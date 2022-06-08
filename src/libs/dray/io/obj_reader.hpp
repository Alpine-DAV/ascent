// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_OBJ_LOADER_H
#define DRAY_OBJ_LOADER_H

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/rendering/material.hpp>
#include <dray/rendering/texture2d.hpp>
#include <vector>

namespace dray
{

void read_obj (const std::string file_path,
               Array<Vec<float32,3>> &a_verts,
               Array<Vec<int32,3>> &a_indices,
               Array<Vec<float32,2>> &t_coords,
               Array<Vec<int32,3>> &t_indices,
               Array<Material> &a_materials,
               Array<int32> &a_mat_ids,
               std::vector<Texture2d> &textures);

} // namespace dray
#endif
