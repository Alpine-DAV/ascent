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

conduit::Node vert_location(const conduit::Node &domain,
                            const int &index,
                            const std::string &topo_name = "");

conduit::Node element_location(const conduit::Node &domain,
                               const int &index,
                               const std::string &topo_name = "");

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
                            const std::string &topo_name);

conduit::Node binning(const conduit::Node &dataset,
                      conduit::Node &bin_axes,
                      const std::string &reduction_var,
                      const std::string &reduction_op,
                      const double empty_bin_val,
                      const std::string &component);

void ASCENT_API paint_binning(const conduit::Node &binning,
                              conduit::Node &dataset,
                              const std::string field_name = "");

void ASCENT_API binning_mesh(const conduit::Node &binning,
                             conduit::Node &mesh,
                             const std::string field_name = "");

conduit::Node get_state_var(const conduit::Node &dataset,
                            const std::string &var_name);

bool is_scalar_field(const conduit::Node &dataset,
                     const std::string &field_name);

bool has_field(const conduit::Node &dataset, const std::string &field_name);

// topology exists on at least one rank
bool has_topology(const conduit::Node &dataset, const std::string &topo_name);

std::set<std::string> topology_names(const conduit::Node &dataset);

//std::string known_topos(const conduit::Node &dataset);

bool has_component(const conduit::Node &dataset,
                   const std::string &field_name,
                   const std::string &component);

// returns -1 if the component does not exist
// does not check for consistency (i.e, if all the
// domains has the same number of components)
int num_components(const conduit::Node &dataset,
                   const std::string &field_name);

std::string component_name(const conduit::Node &dataset,
                           const std::string &field_name,
                           const int component_id);

std::string
possible_components(const conduit::Node &dataset,
                    const std::string &field_name);

bool is_xyz(const std::string &axis_name);

conduit::Node quantile(const conduit::Node &cdf,
                       const double val,
                       const std::string &interpolation);

// if the field node is empty, we will allocate space
void paint_nestsets(const std::string nestset_name,
                    const std::string topo_name,
                    conduit::Node &dom,
                    conduit::Node &field); // field to paint on

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
