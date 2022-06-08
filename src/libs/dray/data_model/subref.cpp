// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/data_model/subref.hpp>

namespace dray
{
  std::ostream & operator<<(std::ostream &out, const Split<Tensor> &tsplit)
  {
    out << "Split<Tensor>{axis=" << tsplit.axis << ", "
        << (!tsplit.f_lower_t_upper ? "lower" : "upper") << ", "
        << "factor=" << tsplit.factor << "}";
    return out;
  }

  std::ostream & operator<<(std::ostream &out, const Split<Simplex> &ssplit)
  {
    out << "Split<Simplex>{vtx_displaced=" << ssplit.vtx_displaced << ", "
        << "vtx_tradeoff=" << ssplit.vtx_tradeoff << ", "
        << "factor=" << ssplit.factor << "}";
    return out;
  }
}
