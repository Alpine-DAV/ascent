//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_blueprint_architect.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_BLUEPRINT_ARCHITECT
#define ASCENT_BLUEPRINT_ARCHITECT

#include <ascent.hpp>
#include <conduit.hpp>
#include <memory>

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

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------
namespace detail
{
int get_num_vertices(const std::string &shape_type);
}
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------

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

conduit::Node field_max(const conduit::Node &dataset,
                        const std::string &field_name);

conduit::Node field_min(const conduit::Node &dataset,
                        const std::string &field_name);

conduit::Node field_sum(const conduit::Node &dataset,
                        const std::string &field_name);

conduit::Node field_avg(const conduit::Node &dataset,
                        const std::string &field_name);

conduit::Node field_nan_count(const conduit::Node &dataset,
                              const std::string &field_name);

conduit::Node field_inf_count(const conduit::Node &dataset,
                              const std::string &field_name);

conduit::Node field_histogram(const conduit::Node &dataset,
                              const std::string &field,
                              const double &min_val,
                              const double &max_val,
                              const int &num_bins);

conduit::Node field_entropy(const conduit::Node &hist);

conduit::Node field_pdf(const conduit::Node &hist);
conduit::Node field_cdf(const conduit::Node &hist);

conduit::Node global_bounds(const conduit::Node &dataset,
                            const conduit::Node &field_names);

conduit::Node binning(const conduit::Node &dataset,
                      conduit::Node &bin_axes,
                      const std::string &reduction_var,
                      const std::string &reduction_op,
                      const double empty_val,
                      const std::string &component,
                      const std::string &topo_name,
                      const std::string &assoc_str);

void paint_binning(const conduit::Node &binning,
                   conduit::Node &dataset,
                   const std::string &field_name,
                   const std::string &topo_name,
                   const std::string &assoc_str,
                   const double default_value);

void binning_mesh(const conduit::Node &binning,
                  conduit::Node &mesh,
                  const std::string &field_name);

conduit::Node get_state_var(const conduit::Node &dataset,
                            const std::string &var_name);

bool is_scalar_field(const conduit::Node &dataset,
                     const std::string &field_name);

// field exists on at least one rank. Does not check that
// all ranks with that topology have this field(maybe it should).
bool has_field(const conduit::Node &dataset, const std::string &field_name);

// topology exists on at least one rank
bool has_topology(const conduit::Node &dataset, const std::string &topo_name);

std::string known_topos(const conduit::Node &dataset);

std::string known_fields(const conduit::Node &dataset);

bool has_component(const conduit::Node &dataset,
                   const std::string &field_name,
                   const std::string &component);

std::string possible_components(const conduit::Node &dataset,
                                const std::string &field_name);

bool is_xyz(const std::string &axis_name);

conduit::Node quantile(const conduit::Node &cdf,
                       const double val,
                       const std::string &interpolation);

// assumes that the field exists
std::string field_assoc(const conduit::Node &dataset,
                        const std::string &field_name);

// double or float, checks for global consistency
std::string field_type(const conduit::Node &dataset,
                       const std::string &field_name);
//
// double or float
std::string coord_dtype(const std::string &topo_name,
                        const conduit::Node &domain);

// double or float, checks for global consistency
std::string global_coord_dtype(const std::string &topo_name,
                               const conduit::Node &dataset);

// topo_types = [points, uniform, rectilinear, curvilinear, unstructured]
// expects that a topology does exist or else it will return none
void topology_types(const conduit::Node &dataset,
                    const std::string &topo_name,
                    int topo_types[5]);

// assumes that the topology exists, globally checks for constistency
int spatial_dims(const conduit::Node &dataset, const std::string &topo_name);

// get the dimensionality for a topology in a domain
int topo_dim(const std::string &topo_name, const conduit::Node &dom);

// finds then name of a topology using the field name. topology might not
// exist on this rank.
std::string field_topology(const conduit::Node &dataset,
                           const std::string &field_name);

// if the field node is empty, we will allocate space
void paint_nestsets(const std::string nestset_name,
                    conduit::Node &dom,
                    conduit::Node &field); // field to paint on

// get the global topology and association if possible and check them against
// the ones supplied
conduit::Node
final_topo_and_assoc(const conduit::Node &dataset,
                     const conduit::Node &bin_axes,
                     const std::string &topo_name,
                     const std::string &assoc_str);
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
