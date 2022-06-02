// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dispatcher.hpp>
#include <dray/error.hpp>
#include <sstream>

namespace dray
{

namespace detail
{
  void cast_mesh_failed(Mesh *mesh, const char *file, unsigned long long line)
  {
    std::stringstream msg;
    msg<<"Cast of mesh '"<<mesh->type_name()<<"' failed ";
    msg<<"("<<file<<", "<<line<<")\n";
    DRAY_ERROR(msg.str());
  }
  void cast_field_failed(Field *field, const char *file, unsigned long long line)
  {
    std::stringstream msg;
    msg<<"Cast of field '"<<field->type_name()<<"' failed ";
    msg<<"("<<file<<", "<<line<<")\n";
    DRAY_ERROR(msg.str());
  }
}

}
