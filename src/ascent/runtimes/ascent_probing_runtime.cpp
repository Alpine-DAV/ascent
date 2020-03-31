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
#include <thread>

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
 * Assign part of the vis load to the vis nodes.
 */
std::vector<int> load_assignment(const std::vector<float> &sim_estimate, 
                                 const std::vector<float> &vis_estimates,
                                 const std::vector<int> &node_map,
                                 const int image_count,
                                 const int sim_node_count,
                                 const int vis_node_count, 
                                 const double vis_budget,
                                 const int world_rank)
{
    assert(sim_estimate.size() == vis_estimates.size());
    
    std::valarray<float> t_inline(0.f, sim_node_count);
    for (size_t i = 0; i < sim_node_count; i++)
        t_inline[i] = vis_estimates[i] * image_count;
    std::valarray<float> t_intransit(0.f, vis_node_count);
    std::valarray<float> t_sim(sim_estimate.data(), sim_node_count);

    std::vector<int> image_counts_sim(sim_node_count, 0);
    std::vector<int> image_counts_vis(vis_node_count, 0);

    // initially: push all vis load to vis nodes (all intransit)
    for (size_t i = 0; i < sim_node_count; i++)
    {
        int target_vis_node = node_map[i];

        t_intransit[target_vis_node] += t_inline[i];
        t_inline[i] = 0.f;
        image_counts_vis[target_vis_node] += image_count;
    }

    // vis budget of 1 means in transit only (i.e., only vis nodes render)
    if (vis_budget < 1.0)
    {
        // push back load to sim nodes until 
        // intransit time is smaller than max(inline + sim)
        // NOTE: this loop is potentially ineffective w/ higher node counts
        int i = 0;
        std::valarray<float> t_inline_sim = t_inline + t_sim;
        while (t_inline_sim.max() < t_intransit.max()) 
        {
            // always push back to the fastest sim node
            int min_id = std::min_element(begin(t_inline_sim), 
                                            end(t_inline_sim))
                                            - begin(t_inline_sim);

            // find the corresponding vis node 
            int target_vis_node = node_map[min_id];

            if (image_counts_vis[target_vis_node] > 0)
            {
                t_intransit[target_vis_node] -= vis_estimates[min_id];
                image_counts_vis[target_vis_node]--;
            
                t_inline[min_id] += vis_estimates[min_id];
                image_counts_sim[min_id]++;
            }
            else    // we ran out of renderings on this vis node
            {
                std::cout << "=== Ran out of renderings on node " 
                        << target_vis_node << std::endl;
                break;
            }

            // if sim node got all its images back for inline rendering
            // -> throw it out of consideration
            if (image_counts_sim[min_id] == image_count)
                t_inline[min_id] = std::numeric_limits<float>::max() - t_sim[min_id];

            // recalculate inline + sim time
            t_inline_sim = t_inline + t_sim;
            ++i;
            if (i > image_count*sim_node_count)
                ASCENT_ERROR("Error during load distribution.")
        }
    }

    std::vector<int> image_counts_combined(image_counts_sim);
    image_counts_combined.insert(image_counts_combined.end(), 
                                 image_counts_vis.begin(), 
                                 image_counts_vis.end());

    if (world_rank == 0)
    {
        std::cout << "=== image_counts ";
        for (auto &a : image_counts_combined)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    return image_counts_combined;
}

/**
 * Assign sim nodes to vis nodes based on their overall sim+vis times.
 */
std::vector<int> node_assignment(const std::vector<int> &rank_order, 
                                 const std::vector<float> &vis_estimates,
                                 const int vis_node_count)
{
    int sim_node_count = rank_order.size() - vis_node_count;
    std::vector<float> vis_node_cost(vis_node_count, 0.f);
    std::vector<int> map(sim_node_count, -1);

    for (int i = 0; i < sim_node_count; ++i)
    {
        // pick node with lowest cost
        int target_vis_node = std::min_element(vis_node_cost.begin(), vis_node_cost.end())
                              - vis_node_cost.begin();
        // asssign the sim to to the vis node
        map[rank_order[i]] = target_vis_node;
        // adapt the cost on the vis node
        vis_node_cost[target_vis_node] += vis_estimates[rank_order[i]];
    }
    return map;
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


int ibsend_using_schema(const Node &node, int dest, int tag, MPI_Comm comm, 
                        MPI_Request *request) 
{     
    conduit::Schema s_data_compact;
    
    // schema will only be valid if compact and contig
    if( node.is_compact() && node.is_contiguous())
    {
        s_data_compact = node.schema();
    }
    else
    {
        node.schema().compact_to(s_data_compact);
    }
    
    std::string snd_schema_json = s_data_compact.to_json();

    conduit::Schema s_msg;
    s_msg["schema_len"].set(DataType::int64());
    s_msg["schema"].set(DataType::char8_str(snd_schema_json.size()+1));
    s_msg["data"].set(s_data_compact);
    
    // create a compact schema to use
    conduit::Schema s_msg_compact;
    s_msg.compact_to(s_msg_compact);
    
    Node n_msg(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    n_msg["schema_len"].set((int64)snd_schema_json.length());
    n_msg["schema"].set(snd_schema_json);
    n_msg["data"].update(node);

    index_t msg_data_size = n_msg.total_bytes_compact();

    int size;
    char *bsend_buf;
    // block until all messages currently in the buffer have been transmitted
    MPI_Buffer_detach(&bsend_buf, &size);
    // clean up old buffer
    free(bsend_buf);

    MPI_Buffer_attach(malloc(msg_data_size + MPI_BSEND_OVERHEAD), 
                             msg_data_size + MPI_BSEND_OVERHEAD);

    int mpi_error = MPI_Ibsend(const_cast<void*>(n_msg.data_ptr()),
                              static_cast<int>(msg_data_size),
                              MPI_BYTE,
                              dest,
                              tag,
                              comm,
                              request);
    
    if (mpi_error)
        std::cout << "ERROR sending dataset to " << dest << std::endl;
    // CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}

//-----------------------------------------------------------------------------
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const int world_size,
                    const int world_rank,
                    const MPI_Comm sim_comm,
                    const int sim_node_count,
                    const std::vector<double> &my_probing_times,
                    const int max_image_count,
                    conduit::Node &data,
                    const double probing_factor,
                    const double vis_budget = 0.1)
{
    assert(vis_budget >= 0.0 && vis_budget <= 1.0);
    // HACK: hijack the vis_budget for rendering tests
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
    float my_avg_probing_time = 0.f;
    float my_sim_estimate = data["state/sim_time"].to_float();
    // max image count without probing images
    int image_count = max_image_count - int(std::round(probing_factor*max_image_count));

    // nodes with the highest rank are our vis nodes
    if (world_rank >= sim_node_count) 
    {
        is_vis_node = true;
        my_dest_rank = world_rank - sim_node_count;
        my_sim_estimate = 0.f;
    }
    // otherwise we are a sim node
    else if (world_size > 1)
    {
        assert(my_probing_times.size() > 0);
        // my_avg_probing_time is in milliseconds
        my_avg_probing_time = float(std::accumulate(my_probing_times.begin(), 
                                                    my_probing_times.end(), 0.0) 
                                          / my_probing_times.size());

        std::cout << "+++ probing times ";
        for (auto &a : my_probing_times)
            std::cout << a << " ";
        std::cout << std::endl;

        // convert from milliseconds to seconds 
        my_avg_probing_time /= 1000.f;
        // my_vis_estimate = (my_avg_probing_time * max_image_count) / 1000.f;
        std::cout << "~~~ " << my_avg_probing_time 
                  << " sec vis time estimate " 
                  << world_rank << std::endl;
    }

#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(mpi_comm_world);
    std::cout << "~~~ " << my_sim_estimate << " sec sim time estimate " 
              << world_rank << std::endl;

    // gather all simulation time estimates
    std::vector<float> sim_estimates(world_size, 0.f);
    MPI_Allgather(&my_sim_estimate, 1, MPI_FLOAT, 
                  sim_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);
    // gather all visualization time estimates
    std::vector<float> vis_estimates(world_size, 0.f);
    MPI_Allgather(&my_avg_probing_time, 1, MPI_FLOAT, 
                  vis_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);

    auto start = std::chrono::system_clock::now();

    // sort the ranks accroding to sim+vis time estimate
    std::vector<int> rank_order = sort_ranks(sim_estimates, vis_estimates);
    
    // generate mapping between sending and receiving nodes
    // std::vector<int> intransit_map = job_assignment(sim_estimates, vis_estimates,
    //                                                 rank_order, vis_node_count,
    //                                                 vis_budget);

    // assign sim nodes to vis nodes
    std::vector<int> node_map = node_assignment(rank_order, vis_estimates, vis_node_count);

    // DEBUG: OUT
    if (world_rank == 0)
    {
        std::cout << "=== node_map ";
        for (auto &a : node_map)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    // distribute rendering load across sim and vis loads
    std::vector<int> image_counts = load_assignment(sim_estimates, vis_estimates,
                                                    node_map,
                                                    image_count,
                                                    sim_node_count, vis_node_count,
                                                    vis_budget, world_rank);

    // // split comm: one comm per vis node
    // MPI_Group world_group;
    // MPI_Comm_group(mpi_comm_world, &world_group);
    // std::vector<MPI_Group> vis_groups(vis_node_count);
    // std::vector<MPI_Comm> vis_comms(vis_node_count);
    // std::vector< std::vector<int> > assigned_sim_nodes(vis_node_count);

    // for (int i = 0; i < node_map.size(); i++)
    //     assigned_sim_nodes[node_map[i]].push_back(i); 

    // for (int i = 0; i < vis_node_count; i++)
    // {
    //     MPI_Group_incl(world_group, assigned_sim_nodes[i].size(), 
    //                     assigned_sim_nodes[i].data(), &vis_groups[i]);
    //     MPI_Comm_create_group(mpi_comm_world, vis_groups[i], 0, &vis_comms[i]);
    // }

    Node ascent_opts, blank_actions;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_comm_world);
    ascent_opts["actions_file"] = "cinema_actions.yaml";
    ascent_opts["is_probing"] = 0;
    ascent_opts["probing_factor"] = probing_factor;
    Node verify_info;

    if (is_vis_node) // all vis nodes: receive data 
    {
        std::vector<int> image_counts_vis;
        std::vector<int> sending_node_ranks;

        std::cout << "~~~~rank " << world_rank << " / " << my_dest_rank
                    << ": receives extract(s) from " ;
        for (int i = 0; i < node_map.size(); ++i)
        {
            if (node_map[i] == my_dest_rank)
            {   
                std::cout << i << " ";
                // conduit::Node n_curr; // = data.append();
                // datasets.push_back(n_curr);
                image_counts_vis.push_back(max_image_count - image_counts[i]);
                sending_node_ranks.push_back(i);
            }
        }
        std::cout << std::endl;

        // relay::mpi::gather(Node &send_node, Node &recv_node, int root, vis_comms(my_dest_rank));

        int sending_count = int(sending_node_ranks.size());
        // MPI_Request requests[sending_count];
        // MPI_Status statuses[sending_count];
        // std::vector<conduit::Node> datasets(sending_count);

        for (int i = 0; i < sending_count; ++i)
        {
            log_time(start, "- before receive ", world_rank);
            std::cout << "~~~~ receiving " << sending_node_ranks[i] << std::endl;
            conduit::Node dataset;
            int err = relay::mpi::recv_using_schema(dataset, sending_node_ranks[i], 0, 
                                                    mpi_comm_world);
            log_time(start, "--- after receive ", world_rank);
            // std::cout << "MPI ERROR RECV " << err << std::endl;
        // }

        // std::cout << "~~! wait_all_recv " << std::endl;
        // relay::mpi::wait_all_recv(sending_count, requests, statuses);
        // std::cout << "~~! wait_all_recv END " << std::endl;

        // render data
        // int count = 0;
        // for (auto &dataset : datasets)
        // {
            if (conduit::blueprint::mesh::verify(dataset, verify_info))
            {
                int offset = max_image_count - image_counts_vis[i];

                if (image_counts_vis[i] > 0)
                {
                    std::cout   << "~~~~ VIS node " << world_rank << " rendering " 
                                << offset << " - " 
                                << offset + image_counts_vis[i] << std::endl;
                    ascent_opts["image_count"] = image_counts_vis[i];
                    ascent_opts["image_offset"] = offset;

                    Ascent ascent_render;
                    ascent_render.open(ascent_opts);
                    ascent_render.publish(dataset);    

                    log_time(start, "+ before render vis ", world_rank);
                    ascent_render.execute(blank_actions);
                    ascent_render.close();
                    log_time(start, "+++ after render vis ", world_rank);
                }
            }
            else
            {
                std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                            << std::endl;
            }
            // ++count;
        }
    }
    else // all sim nodes: send extract to vis nodes
    {
        const int destination = node_map[world_rank] + sim_node_count;
        std::cout << "~~~~rank " << world_rank << ": sends extract to " 
                  <<  node_map[world_rank] + sim_node_count << std::endl;

        // in line rendering using ascent and cinema
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            MPI_Request request;
            log_time(start, " before send ", world_rank);
            ibsend_using_schema(data, destination, 0, mpi_comm_world, &request);
            log_time(start, " after send ", world_rank);

            if (image_counts[world_rank] > 0)
            {
                std::cout   << "~~~~ SIM node " << world_rank << " rendering " 
                            << 0 << " - " 
                            << image_counts[world_rank] << std::endl;
                ascent_opts["image_count"] = image_counts[world_rank];
                ascent_opts["image_offset"] = 0;

                Ascent ascent_render;
                ascent_render.open(ascent_opts);
                ascent_render.publish(data);    // sync happens here

                log_time(start, "+ before render sim ", world_rank);
                ascent_render.execute(blank_actions);
                ascent_render.close();
                log_time(start, "+++ after render sim ", world_rank);
            }
        }
        else
        {
            std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                        << std::endl;
        }

        // conduit::relay::mpi::wait_send(&request, MPI_STATUS_IGNORE);
        // MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

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
                    if (probing_factor < 0 || probing_factor > 1)
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
                // TODO: clean up this mess
                conduit::Node scenes;
                scenes.append() = action["scenes"];
                conduit::Node renders;
                renders.append() = scenes.child(0).child(0)["renders"];
                phi = renders.child(0).child(0)["phi"].to_int();
                theta = renders.child(0).child(0)["theta"].to_int();
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
    // NOTE: we could check for data size instead (sim sends empty data on vis nodes)
    if (world_rank < rank_split && probing_factor > 0.0)
    {
        auto start = std::chrono::system_clock::now();
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime
        ascent_opt["is_probing"] = 1;
        ascent_opt["probing_factor"] = probing_factor;
        ascent_opt["image_count"] = phi * theta;
        ascent_opt["image_offset"] = 0;

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
        log_time(start, "probing ", world_rank);
    }
    else
    {
        render_times.push_back(100.f); // add dummy value for in transit only
    }

#if ASCENT_MPI_ENABLED
    if (probing_factor < 1.0)
    {
        splitAndRender(mpi_comm_world, world_size, world_rank, sim_comm, rank_split, 
                    render_times, phi*theta, m_data, probing_factor, vis_budget);
    }

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
