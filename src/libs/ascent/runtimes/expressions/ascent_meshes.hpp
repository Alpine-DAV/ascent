#ifndef ASCENT_MESHES_HPP
#define ASCENT_MESHES_HPP

#include "ascent_memory_manager.hpp"
#include "ascent_memory_interface.hpp"
#include "ascent_array.hpp"
#include "ascent_execution_policies.hpp"

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

template<typename T, int S>
class Vec
{
public:
  T m_data[S];

  ASCENT_EXEC const T &operator[] (const int &i) const
  {
    return m_data[i];
  }

  ASCENT_EXEC T &operator[] (const int &i)
  {
    return m_data[i];
  }
};

ASCENT_EXEC
static int num_indices(const int shape_id)
{
  int indices = 0;
  if(shape_id == 5)
  {
    indices = 3;
  }
  else if(shape_id == 9)
  {
    indices = 4;
  }
  else if(shape_id == 10)
  {
    indices = 4;
  }
  else if(shape_id == 12)
  {
    indices = 8;
  }
  else if(shape_id == 1)
  {
    indices = 1;
  }
  else if(shape_id == 3)
  {
    indices = 2;
  }
  return indices;
}


ASCENT_EXEC
void
mesh_logical_index_2d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = vert_index / dims[0];
}

ASCENT_EXEC
void
mesh_logical_index_3d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = (vert_index / dims[0]) % dims[1];
  idx[2] = vert_index / (dims[0] * dims[1]);
}

ASCENT_EXEC
void structured_cell_indices(const int cell_index,
                             const Vec<int,3> &point_dims,
                             const int dims, // 2d or 3d
                             int indices[8])
{
  const int element_dims[3] = {point_dims[0]-1,
                               point_dims[1]-1,
                               point_dims[2]-1};
  int element_index[3];
  if(dims == 2)
  {
    mesh_logical_index_2d(element_index, cell_index, element_dims);

    indices[0] = element_index[1] * point_dims[0] + element_index[0];
    indices[1] = indices[0] + 1;
    indices[2] = indices[1] + point_dims[0];
    indices[3] = indices[2] - 1;
  }
  else
  {
    mesh_logical_index_3d(element_index, cell_index, element_dims);

    indices[0] =
        (element_index[2] * point_dims[1] + element_index[1])
         * point_dims[0] + element_index[0];
    indices[1] = indices[0] + 1;
    indices[2] = indices[1] + point_dims[1];
    indices[3] = indices[2] - 1;
    indices[4] = indices[0] + point_dims[0] * point_dims[2];
    indices[5] = indices[4] + 1;
    indices[6] = indices[5] + point_dims[1];
    indices[7] = indices[6] - 1;
  }
}

struct UniformMesh
{
  Vec<int,3> m_point_dims;
  Vec<double,3> m_origin;
  Vec<double,3> m_spacing;
  int m_dims;
  int m_num_cells;
  int m_num_indices;
  int m_num_points;

  UniformMesh() = delete;
  UniformMesh(const conduit::Node &n_coords)
  {
    const conduit::Node &n_dims = n_coords["dims"];
    // assume we have a valid dataset
    m_point_dims[0] = n_dims["i"].to_int();
    m_point_dims[1] = n_dims["j"].to_int();
    m_point_dims[2] = 1;
    m_dims = 2;
    // check for 3d
    if(n_dims.has_path("k"))
    {
      m_point_dims[2] = n_dims["k"].to_int();
      m_dims = 3;
    }

    m_origin[0] = 0.0;
    m_origin[1] = 0.0;
    m_origin[2] = 0.0;

    m_spacing[0] = 1.0;
    m_spacing[1] = 1.0;
    m_spacing[2] = 1.0;

    if(n_coords.has_child("origin"))
    {
      const conduit::Node &n_origin = n_coords["origin"];

      if(n_origin.has_child("x"))
      {
        m_origin[0] = n_origin["x"].to_float64();
      }

      if(n_origin.has_child("y"))
      {
        m_origin[1] = n_origin["y"].to_float64();
      }

      if(n_origin.has_child("z"))
      {
        m_origin[2] = n_origin["z"].to_float64();
      }
    }

    if(n_coords.has_path("spacing"))
    {
      const conduit::Node &n_spacing = n_coords["spacing"];

      if(n_spacing.has_path("dx"))
      {
        m_spacing[0] = n_spacing["dx"].to_float64();
      }

      if(n_spacing.has_path("dy"))
      {
        m_spacing[1] = n_spacing["dy"].to_float64();
      }

      if(n_spacing.has_path("dz"))
      {
        m_spacing[2] = n_spacing["dz"].to_float64();
      }
    }

    m_num_points = m_point_dims[0] * m_point_dims[1];
    m_num_cells = (m_point_dims[0] - 1) *(m_point_dims[1] - 1);
    if(m_dims == 3)
    {
      m_num_cells *= m_point_dims[2] - 1;
      m_num_points = m_point_dims[2];
    }

    if(m_dims == 3)
    {
      m_num_indices = 8;
    }
    else
    {
      m_num_indices = 4;
    }


  }

  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    int logical_idx[3];
    int pdims[3] = {m_point_dims[0],
                    m_point_dims[1],
                    m_point_dims[2]};
    if(m_dims == 2)
    {
      mesh_logical_index_2d(logical_idx, vert_id, pdims);
    }
    else
    {
      mesh_logical_index_3d(logical_idx, vert_id, pdims);
    }

    vert[0] = m_origin[0] + logical_idx[0] * m_spacing[0];
    vert[1] = m_origin[1] + logical_idx[1] * m_spacing[1];
    if(m_dims == 3)
    {
      vert[2] = m_origin[2] + logical_idx[2] * m_spacing[2];
    }
    else
    {
      vert[2] = 0.;
    }

  }

};


template<typename CoordsType>
struct StructuredMesh
{

  MemoryAccessor<CoordsType> m_coords_x;
  MemoryAccessor<CoordsType> m_coords_y;
  MemoryAccessor<CoordsType> m_coords_z;
  const int m_dims;
  const Vec<int,3> m_point_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  StructuredMesh() = delete;
  StructuredMesh(const std::string mem_space,
                 MemoryInterface<CoordsType> &coords,
                 const int dims,
                 const int point_dims[3])
    : m_coords_x(coords.accessor(mem_space, "x")),
      m_coords_y(coords.accessor(mem_space, "y")),
      m_coords_z(dims == 3 ? coords.accessor(mem_space, "z") :
                             // just use a dummy in this case
                             coords.accessor(mem_space, "x")),
      m_dims(dims),
      m_point_dims({{point_dims[0],
                     point_dims[1],
                     point_dims[2]}}),
      m_num_indices(m_dims == 2 ? 4 : 8),
      m_num_cells(m_dims == 2 ? (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1)
                              : (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1) *
                                (m_point_dims[2] - 1)),
      m_num_points(m_dims == 2 ? (m_point_dims[0]) *
                                 (m_point_dims[1])
                               : (m_point_dims[0]) *
                                 (m_point_dims[1]) *
                                 (m_point_dims[2]))
  {
  }

  // TODO: some sort of error checking mechinism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    vert[0] = static_cast<double>(m_coords_x[vert_id]);
    vert[1] = static_cast<double>(m_coords_y[vert_id]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[vert_id]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};

template<typename CoordsType>
struct RectilinearMesh
{

  MemoryAccessor<CoordsType> m_coords_x;
  MemoryAccessor<CoordsType> m_coords_y;
  MemoryAccessor<CoordsType> m_coords_z;
  const int m_dims;
  const Vec<int,3> m_point_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  RectilinearMesh() = delete;
  RectilinearMesh(const std::string mem_space,
                  MemoryInterface<CoordsType> &coords,
                  const int dims)
    : m_coords_x(coords.accessor(mem_space, "x")),
      m_coords_y(coords.accessor(mem_space, "y")),
      m_coords_z(dims == 3 ? coords.accessor(mem_space, "z") :
                             // just use a dummy in this case
                             coords.accessor(mem_space, "x")),
      m_dims(dims),
      m_point_dims({{(int)m_coords_x.m_size,
                     (int)m_coords_y.m_size,
                     (int)m_coords_z.m_size}}),
      m_num_indices(m_dims == 2 ? 4 : 8),
      m_num_cells(m_dims == 2 ? (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1)
                              : (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1) *
                                (m_point_dims[2] - 1)),
      m_num_points(m_dims == 2 ? (m_point_dims[0]) *
                                 (m_point_dims[1])
                               : (m_point_dims[0]) *
                                 (m_point_dims[1]) *
                                 (m_point_dims[2]))
  {
  }

  // TODO: some sort of error checking mechinism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    int logical_idx[3];
    int pdims[3] = {m_point_dims[0],
                    m_point_dims[1],
                    m_point_dims[2]};
    if(m_dims == 2)
    {
      mesh_logical_index_2d(logical_idx, vert_id, pdims);
    }
    else
    {
      mesh_logical_index_3d(logical_idx, vert_id, pdims);
    }

    vert[0] = static_cast<double>(m_coords_x[logical_idx[0]]);
    vert[1] = static_cast<double>(m_coords_y[logical_idx[1]]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[logical_idx[2]]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};

template<typename CoordsType, typename ConnType>
struct UnstructuredMesh
{

  MemoryAccessor<CoordsType> m_coords_x;
  MemoryAccessor<CoordsType> m_coords_y;
  MemoryAccessor<CoordsType> m_coords_z;
  MemoryAccessor<ConnType> m_conn;
  const int m_shape_type;
  const int m_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  UnstructuredMesh() = delete;
  UnstructuredMesh(const std::string mem_space,
                   MemoryInterface<CoordsType> &coords,
                   MemoryInterface<ConnType> &conn,
                   const int shape_type,
                   const int dims)
    : m_coords_x(coords.accessor(mem_space, "x")),
      m_coords_y(coords.accessor(mem_space, "y")),
      m_coords_z(dims == 3 ? coords.accessor(mem_space, "z") :
                             // just use a dummy in this case
                             coords.accessor(mem_space, "x")),
      m_conn(conn.accessor(mem_space)),
      m_shape_type(shape_type),
      m_dims(dims),
      m_num_indices(num_indices(shape_type)),
      m_num_cells(m_conn.m_size / m_num_indices),
      m_num_points(m_coords_x.m_size)

  {
  }

  // TODO: some sort of error checking mechinism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    const int offset = cell_index * m_num_indices;
    for(int i = 0; i < m_num_indices; ++i)
    {
      indices[i] = static_cast<int>(m_conn[offset + i]);
    }
  }

  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    vert[0] = static_cast<double>(m_coords_x[vert_id]);
    vert[1] = static_cast<double>(m_coords_y[vert_id]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[vert_id]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
