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
