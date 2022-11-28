// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_HPP
#define DRAY_HPP

namespace dray
{

class dray
{
  public:
    static void about();
    static void init();
    static void finalize();

    static bool mpi_enabled();
    static int  mpi_size();
    static int  mpi_rank();
    static void mpi_comm(int mpi_comm_id);
    static int  mpi_comm();

    static bool cuda_enabled();
    static bool hip_enabled();

    static void set_face_subdivisions(const int num_subdivions);
    static void set_zone_subdivisions(const int num_subdivions);

    static int get_face_subdivisions();
    static int get_zone_subdivisions();

    // attempt to load fast paths
    // if false, default to general order path
    static void prefer_native_order_mesh(bool on);
    static bool prefer_native_order_mesh();
    static void prefer_native_order_field(bool on);
    static bool prefer_native_order_field();

    static void set_host_allocator_id(int id);
    static void set_device_allocator_id(int id);

  private:
    static int m_face_subdivisions;
    static int m_zone_subdivisions;
    static bool m_prefer_native_order_mesh;
    static bool m_prefer_native_order_field;
};

} // namespace dray
#endif
