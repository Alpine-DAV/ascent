// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MFEM_READER_HPP
#define DRAY_MFEM_READER_HPP

#include <dray/data_model/data_set.hpp>
#include <dray/import_order_policy.hpp>

namespace dray
{

class MFEMReader
{
  public:
  static Collection load(const std::string &root_file,
                         const int cycle,
                         const ImportOrderPolicy &);

  static Collection load(const std::string &root_file,
                         const ImportOrderPolicy & import_order_policy)
  {
    return load(root_file, 0, import_order_policy);
  }
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP
