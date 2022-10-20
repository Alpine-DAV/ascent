// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BLUEPRINT_LOW_ORDER_HPP
#define DRAY_BLUEPRINT_LOW_ORDER_HPP

#include <conduit.hpp>
#include <dray/data_model/collection.hpp>
//#include <dray/data_model/grid_function.hpp>

namespace dray
{

class BlueprintLowOrder
{
public:

  static DataSet import(const conduit::Node &n_dataset);
  static
  std::shared_ptr<Mesh> import_uniform(const conduit::Node &n_coords,
                                       Array<int32> &conn,
                                       int32 &n_elems,
                                       std::string &shape);


  static
  std::shared_ptr<Mesh> import_explicit(const conduit::Node &n_coords,
                                        const conduit::Node &n_topo,
                                        Array<int32> &conn,
                                        int32 &n_elems,
                                        std::string &shape);

  static
  std::shared_ptr<Mesh> import_structured(const conduit::Node &n_coords,
                                          const conduit::Node &n_topo,
                                          Array<int32> &conn,
                                          int32 &n_elems,
                                          std::string &shape);

  static void to_blueprint(const conduit::Node &dray_rep, conduit::Node &n);
};

} // namespace dray

#endif
