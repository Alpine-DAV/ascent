// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DRAY_NODE_TO_DATASET_HPP
#define DRAY_DRAY_NODE_TO_DATASET_HPP

#include <dray/data_model/data_set.hpp>
#include <conduit.hpp>


namespace dray
{

DataSet to_dataset(const conduit::Node &n_dataset);

} // namespace dray

#endif
