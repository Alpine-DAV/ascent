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
                            const std::string topo_name = "");

conduit::Node element_location(const conduit::Node &domain,
                               const int &index,
                               const std::string topo_name = "");

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
                              const std::string &field_name,
                              const double &min_val,
                              const double &max_val,
                              const int &num_bins);

conduit::Node field_entropy(const conduit::Node &hist);

conduit::Node field_pdf(const conduit::Node &hist);
conduit::Node field_cdf(const conduit::Node &hist);

conduit::Node get_state_var(const conduit::Node &dataset,
                            const std::string &var_name);

bool is_scalar_field(const conduit::Node &dataset,
                     const std::string &field_name);

// field exists on at least one rank. Does not check that
// all ranks with that topology have this field(maybe it should).
bool has_field(const conduit::Node &dataset,
               const std::string &field_name);

// topology exists on at least one rank
bool has_topology(const conduit::Node &dataset,
                  const std::string &topo_name);

conduit::Node quantile(const conduit::Node &cdf,
                       const double val,
                       const std::string interpolation);

// assumes that the field exists
std::string field_assoc(const conduit::Node &dataset,
                        const std::string &field_name);
// double or float, checks for global consistency
std::string field_type(const conduit::Node &dataset,
                       const std::string &field_name);

// topo_types = [points, uniform, rectilinear, curvilinear, unstructured]
// expects that a topology does exist or else it will return none
void topology_types(const conduit::Node &dataset,
                    const std::string &topo_name,
                    int topo_types[5]);

// assumes that the topology exists
int num_cells(const conduit::Node &domain, const std::string &topo_name);
// assumes that the topology exists
int num_points(const conduit::Node &domain, const std::string &topo_name);

// assumes that the topology exists, globally checks for constistency
int spatial_dims(const conduit::Node &dataset, const std::string &topo_name);

// finds then name of a topology using the field name. topology might not
// exist on this rank.
std::string field_topology(const conduit::Node &dataset, const std::string &field_name);

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
