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
/// file: ascent_probing_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_probing_runtime.hpp"

// hola
#include <ascent_hola.hpp>
#include <ascent_hola_mpi.hpp>

// standard lib includes
#include <string.h>
#include <cassert>
#include <numeric>
#include <cmath>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/Error.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#endif // ASCENT_VTKM_ENABLED

using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ProbingRuntime::ProbingRuntime()
    : Runtime()
{
}

//-----------------------------------------------------------------------------
ProbingRuntime::~ProbingRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the ascent interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ProbingRuntime::Initialize(const conduit::Node &options)
{
#if ASCENT_MPI_ENABLED
    if (!options.has_child("mpi_comm") ||
        !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::open options missing MPI communicator (mpi_comm)");
    }
#endif
    // check for probing options (?)

    m_runtime_options = options;
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Info(conduit::Node &out)
{
    out.reset();
    out["runtime/type"] = "probing";
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Cleanup()
{
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Publish(const conduit::Node &data)
{
    Node verify_info;
    bool verify_ok = conduit::blueprint::mesh::verify(data, verify_info);

#if ASCENT_MPI_ENABLED

    MPI_Comm mpi_comm = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());

    // parallel reduce to find if there were any verify errors across mpi tasks
    // use an mpi sum to check if all is ok
    // Node n_src, n_reduce;

    // if (verify_ok)
    //     n_src = (int)0;
    // else
    //     n_src = (int)1;

    // conduit::relay::mpi::sum_all_reduce(n_src,
    //                                     n_reduce,
    //                                     mpi_comm);

    // int num_failures = n_reduce.value();
    // if (num_failures != 0)
    // {
    //     ASCENT_ERROR("Mesh Blueprint Verify failed on "
    //                  << num_failures
    //                  << " MPI Tasks");

    //     // you could use mpi to find out where things went wrong ...
    // }

#else
    if (!verify_ok)
    {
        ASCENT_ERROR("Mesh Blueprint Verify failed!"
                     << std::endl
                     << verify_info.to_json());
    }
#endif

    // create our own tree, with all data zero copied.
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------
bool decide_intransit(const std::vector<float> &times,
                      const int world_rank,
                      const float vis_budget)
{
    // TODO: calculate based on budget and sim times
    // double max_time = 120.f;

    if (times.at(world_rank) > vis_budget)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * 
 */
std::vector<int> job_assignment(const std::vector<float> &sim_estimate, 
                                const std::vector<float> &vis_estimates,
                                const std::vector<int> &rank_order, 
                                const int vis_node_count, 
                                const double vis_budget)
{
    assert(sim_estimate.size() == vis_estimates.size() == rank_order.size());
    std::vector<int> map(rank_order.size(), -1);

    // vis_budget of 0 implies in line rendering only
    if (vis_budget <= 0.0 + std::numeric_limits<double>::epsilon())
        return map;

    // TODO: at the moment every vis node gets at least one package 
    // -> no transfer overhead included yet
    std::vector<float> sum(vis_node_count, 0.f);
    
    // loop over sorted ranks excluding vis nodes
    for (int i,j = 0; i < rank_order.size() - vis_node_count; ++i, ++j)
    {
        int vis_node = j % vis_node_count;
        // vis budget of 1 implies in transit only
        if (vis_estimates[rank_order[i]] + sim_estimate[rank_order[i]] > sum[vis_node] 
            || vis_budget >= 1.0 - std::numeric_limits<double>::epsilon()) 
        {
            // assign to vis node
            map[rank_order[i]] = vis_node;
            sum[vis_node] += vis_estimates[rank_order[i]];
        }
    }
    return map;
}

/**
 * Sort ranks in descending order according to sim + vis times estimations.
 * TODO: add transfer overhead
 */
std::vector<int> sort_ranks(const std::vector<float> &sim_estimates, 
                            const std::vector<float> &vis_estimates)
{
    assert(sim_estimates.size() == vis_estimates.size());
    std::vector<int> rank_order(sim_estimates.size());
    std::iota(rank_order.begin(), rank_order.end(), 0);

    std::stable_sort(rank_order.begin(), 
                     rank_order.end(), 
                     [&](int i, int j) 
                     { 
                         return sim_estimates[i] + vis_estimates[i] 
                              > sim_estimates[j] + vis_estimates[j];
                     } 
                     );
    return rank_order;
}

//-----------------------------------------------------------------------------
std::string get_timing_file_name(const int value, const int precision)
{
    std::ostringstream oss;
    oss << "timings/vis_";
    oss << std::setw(precision) << std::setfill('0') << value;
    oss << ".txt";
    return oss.str();
}

//-----------------------------------------------------------------------------
void log_time(std::chrono::time_point<std::chrono::system_clock> start, 
              const std::string &description,
              const int rank)
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count()
    //           << "s rank " << rank << std::endl;
    std::ofstream out(get_timing_file_name(rank, 5), std::ios_base::app);
    out << description << elapsed.count() << std::endl;
    out.close();
}

//-----------------------------------------------------------------------------
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const int world_size,
                    const int world_rank,
                    const MPI_Comm sim_comm,
                    const int sim_node_count,
                    const std::vector<double> &my_probing_times,
                    const int cinema_image_count,
                    conduit::Node &data,
                    const double vis_budget = 0.1)
{
    assert(vis_budget >= 0.0 && vis_budget <= 1.0);
    // TODO: hijack the vis_budget for rendering tests
    // vis_budget of 0.0 => all in line; vis_budget of 1.0 => all in transit

    assert(sim_node_count > 0 && sim_node_count <= world_size);

    int is_intransit = 0;
    int is_rendering = 0;
    int is_sending = 0;
    bool is_vis_node = false;

    int my_src_rank = -1;
    int my_dest_rank = -1;

    int vis_node_count = world_size - sim_node_count;
    float my_vis_estimate = 0.f;
    float my_sim_estimate = data["state/sim_time"].to_float();

    // nodes with the highest rank are our vis nodes
    if (world_rank >= sim_node_count) 
    {
        is_vis_node = true;
        my_dest_rank = world_rank - sim_node_count;
        my_sim_estimate = 0.f;
    }
    // otherwise we are a sim nodes
    else if (world_size > 1)
    {
        assert(my_probing_times.size() > 0);
        double my_avg_probing_time = std::accumulate(my_probing_times.begin(), 
                                                     my_probing_times.end(), 0.0) 
                                     / my_probing_times.size();
        my_vis_estimate = float(my_avg_probing_time * cinema_image_count) / 1000.f;
        std::cout << "~~~ " << my_vis_estimate 
                  << " sec vis time estimate " 
                  << world_rank << std::endl;
    }

#ifdef ASCENT_MPI_ENABLED
    // gather all simulation time estimates
    std::cout << "~~~ " << my_sim_estimate << " sec sim time estimate " 
              << world_rank << std::endl;
    std::vector<float> sim_estimates(world_size, 0.f);
    MPI_Allgather(&my_sim_estimate, 1, MPI_FLOAT, 
                  sim_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);
    // gather all visualization time estimates
    std::vector<float> vis_estimates(world_size, 0.f);
    MPI_Allgather(&my_vis_estimate, 1, MPI_FLOAT, 
                  vis_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);

    auto start = std::chrono::system_clock::now();

    // sort the ranks accroding to sim+vis time estimate
    std::vector<int> rank_order = sort_ranks(sim_estimates, vis_estimates);
    // generate mapping between sending and receiving nodes
    std::vector<int> intransit_map = job_assignment(sim_estimates, vis_estimates,
                                                    rank_order, vis_node_count,
                                                    vis_budget);

    // Debug OUT: output in-transit map
    // std::cout << "=== sim_estimates ";
    // for (auto &a : sim_estimates)
    //     std::cout << a << " ";
    // std::cout << std::endl;

    std::vector<int> world_to_src(intransit_map.size(), -1); 
    std::vector<int> dest_counts(vis_node_count);
    std::vector<int> source_ranks;
    std::vector<int> render_ranks;
    int src_count = 0;
    for (int i = 0; i < intransit_map.size(); i++)
    {
        if (intransit_map[i] >= 0)
        {
            dest_counts[intransit_map[i]]++;
            if (i == world_rank)
            {
                // we are sending -> calc our source rank
                my_src_rank = src_count;
            }
            else if (intransit_map[i] == my_dest_rank)
            {
                // we are a vis node and someone is sending to us
                is_intransit = 1;
                is_rendering = 1;
            }
            // add source to map
            world_to_src[i] = src_count;
            ++src_count;
            source_ranks.push_back(i);
        }
        else
        {
            render_ranks.push_back(i);
        }
    }

    // total number of receiveing nodes
    int dest_count = 0;
    for (auto &a : dest_counts)
        dest_count += (a > 0) ? 1 : 0;

    int intransit_size = src_count + dest_count;
    while (vis_node_count > dest_count)
    {
        render_ranks.pop_back();
        --vis_node_count;
    }

    std::vector<int> intransit_ranks(source_ranks);
    for (size_t i = sim_node_count; i < world_size; i++)
        intransit_ranks.push_back(i);

    // map contains the in transit assignment
    if (intransit_map[world_rank] >= 0)
    {
        // this node is sending in-transit
        is_intransit = 1;
        is_sending = 1;
    }
    else    
    {
        // this node renders in line
        is_rendering = 1;
    }

    // create in-transit group of nodes
    MPI_Group world_group;
    MPI_Comm_group(mpi_comm_world, &world_group);
    MPI_Group intransit_group;
    MPI_Group_incl(world_group, intransit_size, intransit_ranks.data(), &intransit_group);
    MPI_Comm intransit_comm;
    MPI_Comm_create_group(mpi_comm_world, intransit_group, 0, &intransit_comm);
    // MPI_Comm_split(mpi_comm_world, is_intransit, 0, &intransit_comm);

    // Hola setup
    // MPI_Comm hola_comm;
    // MPI_Comm_split(intransit_comm, is_sending, 0, &hola_comm);
    // split world comm into rendering and sending nodes
    MPI_Group render_group;
    MPI_Group_incl(world_group, render_ranks.size(), render_ranks.data(), &render_group);
    MPI_Comm render_comm;
    MPI_Comm_create_group(mpi_comm_world, render_group, 0, &render_comm);

    // MPI_Comm_split(mpi_comm_world, is_rendering, 0, &render_comm); // TODO: sync???

    if (is_rendering && MPI_COMM_NULL != render_comm) // all rendering nodes
    {
        if (is_vis_node) // render on vis node
        {
            std::cout << "~~~~rank " << world_rank << " / " << my_dest_rank
                      << ": receives extract(s) from " ;
            // use hola to receive the extract data
            // conduit::Node hola_opts;
            // hola_opts["mpi_comm"] = MPI_Comm_c2f(intransit_comm);
            // hola_opts["rank_split"] = src_count;
            // ascent::hola("hola_mpi", hola_opts, data);

            // receive data from corresponding sources
            for (int i = 0; i < intransit_map.size(); ++i)
            {
                if (intransit_map[i] == my_dest_rank)
                {   
                    std::cout << i << " ";
                    conduit::Node &n_curr = data.append();
                    int32 src_rank = static_cast<int32>(world_to_src[i]);
                    relay::mpi::recv_using_schema(n_curr,src_rank,0,intransit_comm);
                }
            }
            std::cout << std::endl;
        }
        else // render local (inline)
        {
            std::cout << "~~~~rank " << world_rank << ": renders inline." << std::endl;
        }

        // Full cinema render using Ascent
        Node verify_info;
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            Node ascent_opts, blank_actions;
            // TODO: make the action file name variable
            ascent_opts["actions_file"] = "cinema_actions.yaml";
            ascent_opts["mpi_comm"] = MPI_Comm_c2f(render_comm);

            Ascent ascent_render;
            ascent_render.open(ascent_opts);
            ascent_render.publish(data);    // sync happens here

            log_time(start, "before render ascent execute ", world_rank);
            ascent_render.execute(blank_actions);   // TODO: check for sync
            ascent_render.close();
        }
        else
        {
            std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                      << std::endl;
        }
    }
    else if (!is_vis_node) // all sending nodes: send extract to vis nodes
    {
        // destination as intransit rank
        int destination = intransit_map[world_rank] + src_count;
        std::cout << "~~~~rank " << world_rank << ": sends extract to " 
                  <<  intransit_map[world_rank] + sim_node_count << std::endl;

        relay::mpi::send_using_schema(data, destination, 0, intransit_comm);

        // add the extract
        // conduit::Node actions;
        // conduit::Node &add_extract = actions.append();
        // add_extract["action"] = "add_extracts";
        // add_extract["extracts/e1/type"] = "hola_mpi";
        // add_extract["extracts/e1/params/mpi_comm"] = MPI_Comm_c2f(intransit_comm);
        // add_extract["extracts/e1/params/rank_split"] = sending_node_count;

        // Node ascent_opts;
        // ascent_opts["mpi_comm"] = MPI_Comm_c2f(hola_comm);

        // // Send an extract of the data with Ascent
        // Ascent ascent_send;
        // ascent_send.open(ascent_opts);
        // ascent_send.publish(data);
        // ascent_send.execute(actions); // extract
        // ascent_send.close();
        // std::cout << "----rank " << world_rank << ": ascent_send." << std::endl;
    }

    // clean up the additional comms and groups
    MPI_Group_free(&world_group);
    MPI_Group_free(&render_group);
    // MPI_Comm_free(&render_comm);
    // MPI_Comm_free(&hola_comm);
    MPI_Group_free(&intransit_group);
    // MPI_Comm_free(&intransit_comm); // Fatal error in PMPI_Comm_free: Invalid communicator

    log_time(start, "end splitAndRun ", world_rank);
#endif // ASCENT_MPI_ENABLED
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Execute(const conduit::Node &actions)
{
    int world_rank = 0;
    int world_size = 1;
#if ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm_world = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());
    MPI_Comm_rank(mpi_comm_world, &world_rank);
    MPI_Comm_size(mpi_comm_world, &world_size);
#endif // ASCENT_MPI_ENABLED

    // copy options and actions for probing run
    conduit::Node ascent_opt = m_runtime_options;
    conduit::Node probe_actions = actions;
    // probing setup
    double probing_factor = 0.0;
    double vis_budget = 0.0;
    double node_split = 0.0;
    // cinema angle counts
    int phi = 1;
    int theta = 1;

    // Loop over the actions
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        string action_name = action["action"].as_string();

        if (action_name == "add_scenes")
        {
            if (action.has_path("probing"))
            {
                if (action["probing"].has_path("factor"))
                {
                    probing_factor = action["probing/factor"].to_double();
                    if (probing_factor <= 0 || probing_factor > 1)
                        ASCENT_ERROR("action 'probing': 'probing_factor' must be in range [0,1]");
                }
                else
                {
                    ASCENT_ERROR("action 'probing' missing child 'factor'");
                }

                if (action["probing"].has_path("vis_budget"))
                    vis_budget = action["probing/vis_budget"].to_double();
                else
                    ASCENT_ERROR("action 'probing' missing child 'vis_budget'");

                if (action["probing"].has_path("node_split"))
                {
                    node_split = action["probing/node_split"].to_double();
                    if (node_split <= 0 || node_split > 1)
                        ASCENT_ERROR("action 'probing': 'node_split' must be in range [0,1]");
                }
                else
                {
                    ASCENT_ERROR("action 'probing' missing child 'node_split'");
                }
            }
            else
            {
                ASCENT_ERROR("missing action 'probing'");
            }

            if (action.has_path("scenes"))
            {
                // TODO: clean up this mess (deadlock if action files don't align?)
                conduit::Node scenes;
                scenes.append() = action["scenes"];
                conduit::Node renders;
                renders.append() = scenes.child(0).child(0)["renders"];
                phi = renders.child(0).child(0)["phi"].to_int();
                theta = renders.child(0).child(0)["theta"].to_int();

                // update angle count for probing run
                int phi_probe = int(std::round(phi * probing_factor));
                int theta_probe = int(std::round(theta * probing_factor));
                probe_actions.child(i)["scenes"].child(0)["renders"].child(0)["phi"] = phi_probe;
                probe_actions.child(i)["scenes"].child(0)["renders"].child(0)["theta"] = theta_probe;
            }
            else
            {
                ASCENT_ERROR("action 'add_scenes' missing child 'scenes'");
            }
        }
    }

    int rank_split = 0;
#if ASCENT_MPI_ENABLED
    rank_split = int(std::round(world_size * node_split));
    int color = 0;
    if (world_rank >= rank_split)
        color = 1;

    // construct simulation comm
    MPI_Group world_group;
    MPI_Comm_group(mpi_comm_world, &world_group);
    MPI_Group sim_group;
    std::vector<int> sim_ranks(rank_split);
    std::iota(sim_ranks.begin(), sim_ranks.end(), 0);
    MPI_Group_incl(world_group, rank_split, sim_ranks.data(), &sim_group);
    MPI_Comm sim_comm;
    MPI_Comm_create_group(mpi_comm_world, sim_group, 0, &sim_comm);
    ascent_opt["mpi_comm"] = MPI_Comm_c2f(sim_comm);
#endif // ASCENT_MPI_ENABLED

    std::vector<double> render_times;
    // run probing only if this is a sim node (vis nodes don't have data yet)
    // TODO: we could check for data size instead (sim sends empty data on vis nodes)
    if (world_rank < rank_split)
    {
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime

        // all sim nodes run probing in a new ascent instance
        Ascent ascent_probing;
        ascent_probing.open(ascent_opt);
        ascent_probing.publish(m_data);        // pass on data pointer
        ascent_probing.execute(probe_actions); // pass on actions

        conduit::Node info;
        ascent_probing.info(info);
        NodeIterator itr = info["render_times"].children();
        while (itr.has_next())
        {
            Node &t = itr.next();
            render_times.push_back(t.to_double());
        }
        ascent_probing.close();
    }

#if ASCENT_MPI_ENABLED
    // split comm into sim and vis nodes and render on the respective nodes
    splitAndRender(mpi_comm_world, world_size, world_rank, sim_comm, rank_split, 
                   render_times, phi*theta, m_data, vis_budget);

    MPI_Group_free(&world_group);
    MPI_Group_free(&sim_group);
    // MPI_Comm_free(&sim_comm); // Fatal error in PMPI_Comm_free: Invalid communicator
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
