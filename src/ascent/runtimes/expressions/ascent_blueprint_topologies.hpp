#ifndef ASCENT_BLUEPRINT_TOPOLOGIES
#define ASCENT_BLUEPRINT_TOPOLOGIES

#include <ascent.hpp>
#include <conduit.hpp>
#include <memory>
#include <unordered_set>
#include <array>

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
// -- begin ascent::runtime::expressions --
//-----------------------------------------------------------------------------
namespace expressions
{

class Topology
{

public:
  Topology(const std::string &topo_name,
           const conduit::Node &domain,
           const size_t num_dims);

  virtual ~Topology()
  {
  }
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const = 0;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const = 0;
  virtual size_t get_num_points() const;
  virtual size_t get_num_cells() const;

  const conduit::Node &domain;
  const std::string topo_name;
  const std::string topo_type;
  const std::string coords_name;
  const std::string coords_type;
  const size_t num_dims;

protected:
  size_t num_points;
  size_t num_cells;
};

// T is either float32 or float64
// N is the number of dimensions
template <typename T, size_t N>
class PointTopology : public Topology
{
  static_assert(N >= 1 && N <= 3,
                "Number of Topology dimensions must be between 1 and 3.");

public:
  PointTopology(const std::string &topo_name, const conduit::Node &domain);
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const;
  virtual size_t get_num_cells() const;

private:
  // uniform coords data
  std::array<size_t, N> dims;
  std::array<T, N> origin;
  std::array<T, N> spacing;
  // rectilinear or explicit coords data
  std::array<conduit::DataArray<T>, N> coords;
};

template <typename T, size_t N>
class UniformTopology : public Topology
{
  static_assert(N >= 1 && N <= 3,
                "Number of Topology dimensions must be between 1 and 3.");

public:
  UniformTopology(const std::string &topo_name, const conduit::Node &domain);
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const;

private:
  std::array<size_t, N> dims;
  std::array<T, N> origin;
  std::array<T, N> spacing;
};

template <typename T, size_t N>
class RectilinearTopology : public Topology
{
  static_assert(N >= 1 && N <= 3,
                "Number of Topology dimensions must be between 1 and 3.");

public:
  RectilinearTopology(const std::string &topo_name,
                      const conduit::Node &domain);
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const;

private:
  std::array<conduit::DataArray<T>, N> coords;
};

template <typename T, size_t N>
class StructuredTopology : public Topology
{
  static_assert(N >= 1 && N <= 3,
                "Number of Topology dimensions must be between 1 and 3.");

public:
  StructuredTopology(const std::string &topo_name, const conduit::Node &domain);
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const;

private:
  std::array<size_t, N> dims;
  std::array<conduit::DataArray<T>, N> coords;
};

// TODO only supports single shape topologies
template <typename T, size_t N>
class UnstructuredTopology : public Topology
{
  static_assert(N >= 1 && N <= 3,
                "Number of Topology dimensions must be between 1 and 3.");

public:
  UnstructuredTopology(const std::string &topo_name,
                       const conduit::Node &domain);
  virtual std::array<conduit::float64, 3>
  vertex_location(const size_t index) const;
  virtual std::array<conduit::float64, 3>
  element_location(const size_t index) const;
  virtual size_t get_num_points() const;

private:
  std::array<conduit::DataArray<T>, N> coords;
  conduit::DataArray<conduit::int32> connectivity;
  std::string shape;
  // single shape
  size_t shape_size;
  // polygonal
  conduit::DataArray<conduit::int32> sizes;
  conduit::DataArray<conduit::int32> offsets;
  // polyhedral
  conduit::DataArray<conduit::int32> polyhedral_sizes;
  conduit::DataArray<conduit::int32> polyhedral_offsets;
  conduit::DataArray<conduit::int32> polyhedral_connectivity;
  std::string polyhedral_shape;
  // polyhedra consisting of single shapes
  size_t polyhedral_shape_size;
};

std::unique_ptr<Topology> topologyFactory(const std::string &topo_name,
                                          const conduit::Node &domain);

int get_num_vertices(const std::string &shape_type);
int topo_dim(const std::string &topo_name, const conduit::Node &dom);

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
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
