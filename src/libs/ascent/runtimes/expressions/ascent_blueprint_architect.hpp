//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
// TODO this is temporary
#include <ascent_exports.h>
#include <expressions/ascent_array.hpp>

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

ASCENT_API
conduit::Node vert_location(const conduit::Node &domain,
                            const int &index,
                            const std::string &topo_name = "");

ASCENT_API
conduit::Node element_location(const conduit::Node &domain,
                               const int &index,
                               const std::string &topo_name = "");

ASCENT_API
conduit::Node field_max(const conduit::Node &dataset,
                        const std::string &field_name);

ASCENT_API
conduit::Node field_min(const conduit::Node &dataset,
                        const std::string &field_name);

ASCENT_API
conduit::Node field_sum(const conduit::Node &dataset,
                        const std::string &field_name);

ASCENT_API
conduit::Node field_avg(const conduit::Node &dataset,
                        const std::string &field_name);

ASCENT_API
conduit::Node field_nan_count(const conduit::Node &dataset,
                              const std::string &field_name);

ASCENT_API
conduit::Node field_inf_count(const conduit::Node &dataset,
                              const std::string &field_name);

ASCENT_API
conduit::Node field_histogram(const conduit::Node &dataset,
                              const std::string &field,
                              const double &min_val,
                              const double &max_val,
                              const int &num_bins);

ASCENT_API
conduit::Node histogram_entropy(const conduit::Node &hist);

ASCENT_API
conduit::Node histogram_pdf(const conduit::Node &hist);

ASCENT_API
conduit::Node histogram_cdf(const conduit::Node &hist);

ASCENT_API
conduit::Node global_bounds(const conduit::Node &dataset,
                            const std::string &topo_name);

//
// NOTE: ascent_data_binning contains a RAJA version
// of binning that needs more work, but should eventually
// supercede these versions
// 

ASCENT_API
conduit::Node binning(const conduit::Node &dataset,
                      conduit::Node &bin_axes,
                      const std::string &reduction_var,
                      const std::string &reduction_op,
                      const double empty_bin_val,
                      const std::string &component);

ASCENT_API
void ASCENT_API paint_binning(const conduit::Node &binning,
                              conduit::Node &dataset,
                              const std::string field_name = "");

void ASCENT_API binning_mesh(const conduit::Node &binning,
                             conduit::Node &mesh,
                             const std::string field_name = "");


ASCENT_API
conduit::Node get_state_var(const conduit::Node &dataset,
                            const std::string &var_name);

ASCENT_API
bool is_scalar_field(const conduit::Node &dataset,
                     const std::string &field_name);

ASCENT_API
bool has_field(const conduit::Node &dataset, const std::string &field_name);

// topology exists on at least one rank
ASCENT_API
bool has_topology(const conduit::Node &dataset, const std::string &topo_name);

ASCENT_API
std::set<std::string> topology_names(const conduit::Node &dataset);

//std::string known_topos(const conduit::Node &dataset);

ASCENT_API
bool has_component(const conduit::Node &dataset,
                   const std::string &field_name,
                   const std::string &component);

ASCENT_API
// returns -1 if the component does not exist
// does not check for consistency (i.e, if all the
// domains has the same number of components)
int num_components(const conduit::Node &dataset,
                   const std::string &field_name);

ASCENT_API
std::string component_name(const conduit::Node &dataset,
                           const std::string &field_name,
                           const int component_id);

ASCENT_API
std::string
possible_components(const conduit::Node &dataset,
                    const std::string &field_name);

ASCENT_API
bool is_xyz(const std::string &axis_name);

ASCENT_API
conduit::Node quantile(const conduit::Node &cdf,
                       const double val,
                       const std::string &interpolation);

// if the field node is empty, we will allocate space
ASCENT_API
void paint_nestsets(const std::string nestset_name,
                    const std::string topo_name,
                    conduit::Node &dom,
                    conduit::Node &field); // field to paint on

ASCENT_API
int num_points(const conduit::Node &domain, const std::string &topo_name);

ASCENT_API
int num_cells(const conduit::Node &domain, const std::string &topo_name);

ASCENT_API
bool field_is_float32(const conduit::Node &field);

ASCENT_API
bool field_is_float64(const conduit::Node &field);

ASCENT_API
bool field_is_int32(const conduit::Node &field);

ASCENT_API
bool field_is_int64(const conduit::Node &field);

ASCENT_API
Array<double>
centroids(const conduit::Node &domain, const std::string topo);

ASCENT_API
Array<double>
vertices(const conduit::Node &domain, const std::string topo);

ASCENT_API
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
