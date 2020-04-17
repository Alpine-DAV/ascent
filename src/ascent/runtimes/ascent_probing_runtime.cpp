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

#include <vtkm/filter/VectorMagnitude.h>

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
                                 const int render_count,
                                 const int sim_node_count,
                                 const int vis_node_count, 
                                 const double vis_budget,
                                 const int world_rank)
{
    assert(sim_estimate.size() == vis_estimates.size());
    
    std::valarray<float> t_inline(0.f, sim_node_count);
    for (size_t i = 0; i < sim_node_count; i++)
        t_inline[i] = vis_estimates[i] * render_count;
    std::valarray<float> t_intransit(0.f, vis_node_count);
    std::valarray<float> t_sim(sim_estimate.data(), sim_node_count);

    std::vector<int> render_counts_sim(sim_node_count, 0);
    std::vector<int> render_counts_vis(vis_node_count, 0);

    // initially: push all vis load to vis nodes (=> all intransit case)
    for (size_t i = 0; i < sim_node_count; i++)
    {
        const int target_vis_node = node_map[i];

        t_intransit[target_vis_node] += t_inline[i];
        t_inline[i] = 0.f;
        render_counts_vis[target_vis_node] += render_count;
    }

    // vis budget of 1 implies intransit only (i.e., only vis nodes render)
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
            const int min_id = std::min_element(begin(t_inline_sim), 
                                                end(t_inline_sim))
                                                - begin(t_inline_sim);

            // find the corresponding vis node 
            const int target_vis_node = node_map[min_id];

            if (render_counts_vis[target_vis_node] > 0)
            {
                t_intransit[target_vis_node] -= vis_estimates[min_id];
                render_counts_vis[target_vis_node]--;
            
                t_inline[min_id] += vis_estimates[min_id];
                render_counts_sim[min_id]++;
            }
            else    // we ran out of renderings on this vis node
            {
                std::cout << "=== Ran out of renderings on node " 
                        << target_vis_node << std::endl;
                break;
            }

            // if sim node got all its images back for inline rendering
            // -> throw it out of consideration
            if (render_counts_sim[min_id] == render_count)
                t_inline[min_id] = std::numeric_limits<float>::max() - t_sim[min_id];

            // recalculate inline + sim time
            t_inline_sim = t_inline + t_sim;
            ++i;
            if (i > render_count*sim_node_count)
                ASCENT_ERROR("Error during load distribution.")
        }
    }

    std::vector<int> render_counts_combined(render_counts_sim);
    render_counts_combined.insert(render_counts_combined.end(), 
                                  render_counts_vis.begin(), 
                                  render_counts_vis.end());

    if (world_rank == 0)
    {
        std::cout << "=== render_counts ";
        for (auto &a : render_counts_combined)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    return render_counts_combined;
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
        const int target_vis_node = std::min_element(vis_node_cost.begin(), vis_node_cost.end())
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


int recv_any_using_schema(Node &node, int src, int tag, MPI_Comm comm)
{  
    MPI_Status status;
    
    int mpi_error = MPI_Probe(src, tag, comm, &status);
    
    int buffer_size = 0;
    MPI_Get_count(&status, MPI_BYTE, &buffer_size);

    Node n_buffer(DataType::uint8(buffer_size));
    
    mpi_error = MPI_Recv(n_buffer.data_ptr(),
                         buffer_size,
                         MPI_BYTE,
                         src,
                         tag,
                         comm,
                         &status);

    uint8 *n_buff_ptr = (uint8*)n_buffer.data_ptr();

    Node n_msg;
    // length of the schema is sent as a 64-bit signed int
    // NOTE: we aren't using this value  ... 
    n_msg["schema_len"].set_external((int64*)n_buff_ptr);
    n_buff_ptr +=8;
    // wrap the schema string
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    // create the schema
    Schema rcv_schema;
    Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);

    // advance by the schema length
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    
    // apply the schema to the data
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);
    
    // copy out to our result node
    node.update(n_msg["data"]);
    
    if (mpi_error)
        std::cout << "ERROR receiving dataset from " << status.MPI_SOURCE << std::endl;

    return status.MPI_SOURCE;
}

void pack_node(const Node &node, Node &n_msg)
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
    
    n_msg.set_schema(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    n_msg["schema_len"].set((int64)snd_schema_json.length());
    n_msg["schema"].set(snd_schema_json);
    n_msg["data"].update(node);

    // return n_msg;
}

int 
isend_using_schema(const Node &node, int dest, int tag, MPI_Comm comm, MPI_Request* request)
{     
    Schema s_data_compact;
    
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
        
    Schema s_msg;
    s_msg["schema_len"].set(DataType::int64());
    s_msg["schema"].set(DataType::char8_str(snd_schema_json.size()+1));
    s_msg["data"].set(s_data_compact);
    
    // create a compact schema to use
    Schema s_msg_compact;
    s_msg.compact_to(s_msg_compact);
    
    Node n_msg(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    n_msg["schema_len"].set((int64)snd_schema_json.length());
    n_msg["schema"].set(snd_schema_json);
    n_msg["data"].update(node);

    index_t msg_data_size = n_msg.total_bytes_compact();

    int mpi_error = MPI_Isend(const_cast<void*>(n_msg.data_ptr()),
                             static_cast<int>(msg_data_size),
                             MPI_BYTE,
                             dest,
                             tag,
                             comm,
                             request);
    if (mpi_error)
        std::cout << "ERROR sending dataset to " << dest << std::endl;

    return mpi_error;
}


int ibsend_using_schema(const Node &n_msg, const int dest, const int tag, 
                        MPI_Comm comm, MPI_Request *request) 
{     
    index_t msg_data_size = n_msg.total_bytes_compact();

    int mpi_error = MPI_Bsend(const_cast<void*>(n_msg.data_ptr()),
                               static_cast<int>(msg_data_size),
                               MPI_BYTE,
                               dest,
                               tag,
                               comm);
                            //    request);
    
    if (mpi_error)
        std::cout << "ERROR sending dataset to " << dest << std::endl;

    return mpi_error;
}

/**
 * Detach and free MPI buffer.
 * Blocks until all buffered send messanges got received.
 */
void detach_mpi_buffer()
{
    int size;
    char *bsend_buf;
    // block until all messages currently in the buffer have been transmitted
    MPI_Buffer_detach(&bsend_buf, &size);
    // clean up old buffer
    free(bsend_buf);
}

/**
 * Calculate the message size for sending the render chunks.  
 */
int calc_render_msg_size(const int render_count, const int probing_count, 
                         const double probing_factor, const int width = 1024,
                         const int height = 1024, const int channels = 4+1)
{
    const int inline_renders = render_count - int(probing_factor * render_count);
    const int total_renders = probing_count + inline_renders;
    const int overhead_render = 396;
    const int overhead_global = 288;
    return total_renders * channels * width * height + 
            total_renders * overhead_render + overhead_global;
}

//-----------------------------------------------------------------------------
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const int world_size,
                    const int world_rank,
                    const MPI_Comm sim_comm,
                    const int sim_node_count,
                    const std::vector<double> &my_probing_times,
                    const int max_render_count,
                    conduit::Node &data,
                    conduit::Node &render_chunks_probing,
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
    // max render count without probing renders
    const int probing_count = int(std::round(probing_factor*max_render_count));
    const int render_count = max_render_count - probing_count;

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
        std::cout << world_rank << std::endl;

        // convert from milliseconds to seconds 
        my_avg_probing_time /= 1000.f;
        // my_vis_estimate = (my_avg_probing_time * max_render_count) / 1000.f;
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

    auto start0 = std::chrono::system_clock::now();    

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
    std::vector<int> render_counts = load_assignment(sim_estimates, vis_estimates,
                                                     node_map,
                                                     render_count,
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

    // TODO: remove the magic numbers
    // (# probing images + # inline images) * (RGBA + depth) * width*height + distance cam/dom center
    // const int width = 1024;
    // const int height = 1024;
    // const int channels = 4 + 1; // RGBA + depth
    // const int render_size = (probing_count + render_counts[world_rank]) * channels * width*height + 4;
    // // Node n_msg;
    // // pack_node(data, n_msg);
    // index_t msg_data_size = 10000000; //compact_img.total_bytes_compact();
    // detach_mpi_buffer();
    // // attach new buffer
    // MPI_Buffer_attach(malloc(msg_data_size + render_size + MPI_BSEND_OVERHEAD), 
    //                         msg_data_size + render_size + MPI_BSEND_OVERHEAD);
    // std::cout << "-- buffer size: " << (msg_data_size + render_size + MPI_BSEND_OVERHEAD) << std::endl;


    Node ascent_opts, blank_actions;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_comm_world);
    ascent_opts["actions_file"] = "cinema_actions.yaml";
    ascent_opts["is_probing"] = 0;
    ascent_opts["probing_factor"] = probing_factor;
    Node verify_info;

    if (is_vis_node) // vis nodes 
    {
        std::vector<int> sending_node_ranks;
        for (int i = 0; i < node_map.size(); ++i)
        {
            if (node_map[i] == my_dest_rank)
                sending_node_ranks.push_back(i);
        }

        std::stringstream node_string;
        std::copy(sending_node_ranks.begin(), sending_node_ranks.end(), 
                    std::ostream_iterator<int>(node_string, " "));
        std::cout << "~~~~ vis node " << world_rank << ": receives extract(s) from " 
                  << node_string.str() << std::endl;

        const int sending_count = int(sending_node_ranks.size());

        std::vector<int> srcIds(sending_count);
        std::vector<Node> render_chunks_vis;
        std::vector<Node> datasets;
        // receive all data sets
        for (int i = 0; i < sending_count; ++i)
        {
            auto start = std::chrono::system_clock::now();
            conduit::Node dataset;
            srcIds[i] = recv_any_using_schema(dataset, MPI_ANY_SOURCE, 0, mpi_comm_world);
            std::cout << "~~~ vis node " << world_rank << " received data from "
                      << srcIds[i] << std::endl;
            log_time(start, "- receive data ", world_rank);
            datasets.push_back(dataset);
        }

        // post receives for the render chunks to receive asynchronous while rendering
        std::vector< std::vector<Node>> render_chunks_sim(sending_count);
        std::vector<MPI_Request> requests;
        const int batch_size = 10;
        auto t_start = std::chrono::system_clock::now();        //< timer receive
        for (int i = 0; i < sending_count; ++i)
        {
            int batch_rest = render_counts[srcIds[i]] % batch_size;
            int batch_runs = render_counts[srcIds[i]] / batch_size + 1;

            std::cout << " ~~~ vis node " << world_rank << " receiving " << batch_runs 
                      << " render chunks from " << srcIds[i] << std::endl;

            for (int j = 0; j < batch_runs; ++j)
            {
                const int current_batch_size = (j == batch_runs - 1) ? batch_rest : batch_size;
                if (current_batch_size == 0)
                    break;

                const int buffer_size = calc_render_msg_size(current_batch_size, probing_count, 
                                                             probing_factor);
                Node n_buffer(DataType::uint8(buffer_size));
                render_chunks_sim[i].push_back(n_buffer);
                MPI_Request req = MPI_REQUEST_NULL;
                requests.push_back(req);

                int mpi_error = MPI_Irecv(n_buffer.data_ptr(),
                                          buffer_size,
                                          MPI_BYTE,
                                          srcIds[i],
                                          j+1,
                                          mpi_comm_world,
                                          &req);
                if (mpi_error)
                    std::cout << "ERROR receiving dataset from " << srcIds[i] << std::endl;
            }
        }
        log_time(t_start, "- receive img ", world_rank);        //< timer

        // render all data sets
        for (int i = 0; i < sending_count; ++i)
        {
            if (conduit::blueprint::mesh::verify(datasets[i], verify_info))
            {
                const int current_render_count = max_render_count - render_counts[srcIds[i]];
                const int render_offset = max_render_count - current_render_count;

                if (current_render_count > 0)
                {
                    std::cout   << "~~~~ VIS node " << world_rank << " rendering " 
                                << render_offset << " - " 
                                << render_offset + current_render_count << std::endl;
                    ascent_opts["render_count"] = current_render_count;
                    ascent_opts["render_offset"] = render_offset;
                    ascent_opts["vis_node"] = true;

                    auto start = std::chrono::system_clock::now();
                    Ascent ascent_render;
                    ascent_render.open(ascent_opts);
                    ascent_render.publish(datasets[i]);
                    ascent_render.execute(blank_actions);
                    log_time(start, "+ render vis " + std::to_string(current_render_count) + " ", world_rank);
                    
                    conduit::Node info;
                    ascent_render.info(info);
                    NodeIterator itr = info["render_times"].children();
                    while (itr.has_next())
                    {
                        Node &t = itr.next();
                        // render_times.push_back(t.to_double());
                        std::cout << t.to_double() << " ";
                    }
                    std::cout << "render times VIS node +++ " << world_rank << std::endl;

                    Node render_chunks;
                    render_chunks["depths"] = info["depths"];
                    render_chunks["color_buffers"] = info["color_buffers"];
                    render_chunks["depth_buffers"] = info["depth_buffers"];
                    render_chunks_vis.push_back(render_chunks);

                    ascent_render.close();
                }
            }
            else
            {
                std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                            << std::endl;
            }
        }

        // wait for all render chunks to be received
        t_start = std::chrono::system_clock::now();
        for (MPI_Request req : requests)
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        log_time(t_start, "+ wait receive img ", world_rank);

        // std::vector<Node> render_chunks_sim;
        // std::vector<MPI_Request> requests(sending_count);
        // // recv render chunks
        // for (int i = 0; i < sending_count; ++i)
        // {
        //     std::cout << " ~~~ vis node " << world_rank << " receiving render chunks from "
        //             << srcIds[i] << std::endl;
        //     const int render_size = calc_render_msg_size(render_counts[srcIds[i]], 
        //                                                 probing_count, probing_factor);
        //     // std::cout << "++ estimate " << world_rank << " " << render_size
        //     //             << std::endl;

        //     auto start = std::chrono::system_clock::now();
        //     // receive render chunks from all associated sim nodes
        //     // TODO: replace render_size with actual one from probing
        //     conduit::Node render_chunks(DataType::uint8(render_size));

        //     // int mpi_error = MPI_Recv(render_chunks.data_ptr(), render_size, 
        //     //                          MPI_BYTE, srcIds[i], 1, mpi_comm_world, MPI_STATUS_IGNORE);//, &(requests[i]));
        //     // if (mpi_error)
        //     //     std::cout << "ERROR receiving render chunks from " << srcIds[i] << std::endl;

        //     recv_any_using_schema(render_chunks, srcIds[i], 1, mpi_comm_world);
            
        //     // std::cout << "~~~ vis node " << world_rank << " post irecv RENDER CHUNKS from "
        //     //           << srcIds[i] << std::endl;
        //     render_chunks_sim.push_back(render_chunks);
        //     log_time(start, "- receive img ", world_rank);
        // }

        // receive all render chunks
        // for (int i = 0; i < sending_count; ++i)
        // {
        //     std::cout << " ~~~ vis node " << world_rank << " receiving render chunks from "
        //               << srcIds[i] << std::endl;

        //     auto start = std::chrono::system_clock::now();
        //     // receive render chunks from all associated sim nodes
        //     MPI_Wait(&(requests[i]), MPI_STATUS_IGNORE);

        //     std::cout << " ~~~ vis node " << world_rank << " done" << std::endl;

        //     // unpack
        //     {
        //         uint8 *n_buff_ptr = (uint8*)render_chunks_sim[i].data_ptr();

        //         Node n_msg;
        //         // length of the schema is sent as a 64-bit signed int
        //         // NOTE: we aren't using this value  ... 
        //         n_msg["schema_len"].set_external((int64*)n_buff_ptr);
        //         n_buff_ptr +=8;
        //         // wrap the schema string
        //         n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
        //         // create the schema
        //         Schema rcv_schema;
        //         Generator gen(n_msg["schema"].as_char8_str());
        //         gen.walk(rcv_schema);

        //         // advance by the schema length
        //         n_buff_ptr += n_msg["schema"].total_bytes_compact();
                
        //         // apply the schema to the data
        //         n_msg["data"].set_external(rcv_schema,n_buff_ptr);
                
        //         // copy out to our result node
        //         render_chunks_sim[i].update(n_msg["data"]);
        //     }
        //     log_time(start, "- receive img wait ", world_rank);
        // }

        // {
        //     // TODO: compositing (move to separate method)
        //     vtkh::Scene scene;
        //     vtkh::Renderer *renderer = new vtkh::VolumeRenderer();
        //     scene.AddRenderer(renderer);

        //     std::vector<vtkh::Render> renders;

        //     // unpack render_chunks_sim + render_chunks_vis
        //     for (size_t i = 0; i < count; i++)
        //     {
        //         vtkh::Render r;
        //         // r.SetCamera(const vtkm::rendering::Camera &camera)
        //         renders.push_back(render);
        //     }            
        //     scene.SetRenders(renders);
        //     (*renderer)->Composite(total_renders);
        // }
    } // end vis node
    else // SIM nodes
    {
        const int destination = node_map[world_rank] + sim_node_count;
        std::cout << "~~~~rank " << world_rank << ": sends extract to " 
                  <<  node_map[world_rank] + sim_node_count << std::endl;

        std::vector<Node> compact_img_batches;
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            // send data to vis node
            auto t_start = std::chrono::system_clock::now();

            conduit::relay::mpi::send_using_schema(data, destination, 0, mpi_comm_world);
            // isend_using_schema(data, destination, 0, mpi_comm_world, &req);
            log_time(t_start, "- send data ", world_rank);

            // in line rendering using ascent
            Node render_chunks_inline;
            const int batch_size = 20;
            const int batch_rest = render_counts[world_rank] % batch_size;
            const int batch_runs = render_counts[world_rank] / batch_size + 1;

            std::vector<MPI_Request> requests;
            t_start = std::chrono::system_clock::now();
            for (int i = 0; i < batch_runs; ++i)
            {
                const int begin = i*batch_size;
                const int current_batch_size = (i == batch_runs - 1) ? batch_rest : batch_size;
                if (current_batch_size == 0)
                    break;
                const int end = i*batch_size + current_batch_size;

                std::cout   << "~~~~ SIM node " << world_rank << " rendering " 
                            << begin << " - " << end << std::endl;
                
                ascent_opts["render_count"] = end - begin;
                ascent_opts["image_offset"] = begin;
                ascent_opts["vis_node"] = false;

                Ascent ascent_render;
                ascent_render.open(ascent_opts);
                ascent_render.publish(data);
                ascent_render.execute(blank_actions);

                // send render chunks
                conduit::Node info;
                ascent_render.info(info);
                render_chunks_inline["depths"] = info["depths"];
                render_chunks_inline["color_buffers"] = info["color_buffers"];
                render_chunks_inline["depth_buffers"] = info["depth_buffers"];

                ascent_render.close();

                Node compact_img;
                compact_img_batches.push_back(compact_img); // keep the buffer alive
                pack_node(render_chunks_inline, compact_img); 
                const int render_size = compact_img.total_bytes_compact();
                // const int render_size = calc_render_msg_size(current_batch_size, probing_count, 
                //                                              probing_factor);
                MPI_Request req = MPI_REQUEST_NULL;
                requests.push_back(req);
                int mpi_error = MPI_Isend(const_cast<void*>(compact_img.data_ptr()),
                                          static_cast<int>(render_size),
                                          MPI_BYTE,
                                          destination,
                                          i+1,
                                          mpi_comm_world,
                                          &req);
                if (mpi_error)
                    std::cout << "ERROR sending dataset to " << destination << std::endl;

            }
            log_time(t_start, "+ render sim " + std::to_string(render_counts[world_rank]) + " ", world_rank);

            // wait for all render chunks to be received
            t_start = std::chrono::system_clock::now();
            for (auto req : requests)
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            log_time(t_start, "+ wait send img ", world_rank);

            // TODO: send probing chunks

            // if (render_counts[world_rank] > 0)
            // {
            //     ascent_opts["render_count"] = render_counts[world_rank];
            //     ascent_opts["image_offset"] = 0;
            //     ascent_opts["vis_node"] = false;

            //     t_start = std::chrono::system_clock::now();
            //     Ascent ascent_render;
            //     ascent_render.open(ascent_opts);
            //     ascent_render.publish(data);

            //     ascent_render.execute(blank_actions);
            //     log_time(t_start, "+ render sim " + std::to_string(render_counts[world_rank]) + " ", world_rank);

            //     // send render chunks from in line rendering
            //     conduit::Node info;
            //     ascent_render.info(info);
            //     render_chunks_inline["depths"] = info["depths"];
            //     render_chunks_inline["color_buffers"] = info["color_buffers"];
            //     render_chunks_inline["depth_buffers"] = info["depth_buffers"];

            //     NodeIterator itr = info["render_times"].children();
            //     while (itr.has_next())
            //     {
            //         Node &t = itr.next();
            //         // render_times.push_back(t.to_double());
            //         std::cout << t.to_double() << " ";
            //     }
            //     std::cout << "render times SIM node +++ " << world_rank << std::endl;
            //     ascent_render.close();

            //     // wait for sending data to finish
            //     // MPI_Wait(&req, MPI_STATUS_IGNORE);

            //     // send render chunks from probing and inline rendering
            //     Node render_chunks;
            //     render_chunks["inline"] = render_chunks_inline;
            //     render_chunks["probing"] = render_chunks_probing;

            //     t_start = std::chrono::system_clock::now();
            //     Node compact_img;
            //     pack_node(render_chunks, compact_img);    // this takes too long

            //     const index_t msg_data_size = compact_img.total_bytes_compact();
            //     std::cout << "++ msg_data_size " << world_rank << " " <<  msg_data_size
            //                 << std::endl;

            //     const int render_size = calc_render_msg_size(render_counts[world_rank], probing_count, 
            //                                                 probing_factor);
            //     std::cout << "++ estimate " << world_rank << " " << render_size
            //                 << std::endl;

            //     detach_mpi_buffer();
            //     MPI_Buffer_attach(malloc(msg_data_size + MPI_BSEND_OVERHEAD), 
            //                              msg_data_size + MPI_BSEND_OVERHEAD);
                
            //     conduit::relay::mpi::send_using_schema(compact_img, destination, 1, mpi_comm_world);
            //     // ibsend_using_schema(compact_img, destination, 1, mpi_comm_world, &request);
            //     log_time(t_start, "- send sim img ", world_rank);
            // }
        }
        else
        {
            std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                        << std::endl;
        }
        // conduit::relay::mpi::wait_send(&request, MPI_STATUS_IGNORE);
        // MPI_Wait(&request, MPI_STATUS_IGNORE);
    } // end vis node

    log_time(start0, "___splitAndRun ", world_rank);
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
    // TODO: handle case where there is no probing (and no probing chunks)
    Node render_chunks;
    // run probing only if this is a sim node
    // NOTE: we could check for data size instead (sim sends empty data on vis nodes)
    if (world_rank < rank_split && probing_factor > 0.0)
    {
        auto start = std::chrono::system_clock::now();
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime
        ascent_opt["is_probing"] = 1;
        ascent_opt["probing_factor"] = probing_factor;
        ascent_opt["render_count"] = phi * theta;
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
        
        render_chunks["depths"] = info["depths"];
        render_chunks["color_buffers"] = info["color_buffers"];
        render_chunks["depth_buffers"] = info["depth_buffers"];

        // std::vector<float> depths;
        // itr = info["depths"].children();
        // while (itr.has_next())
        // {
        //     Node &depth = itr.next();
        //     depths.push_back(depth.to_float());
        // }
        // std::vector<Node> color_buffers;
        // itr = info["color_buffers"].children();
        // while (itr.has_next())
        // {
        //     Node &b = itr.next();
        //     color_buffers.push_back(b);
        // }
        // std::vector<Node> depth_buffers;
        // itr = info["depth_buffers"].children();
        // while (itr.has_next())
        // {
        //     Node &b = itr.next();
        //     depth_buffers.push_back(b);
        // }
        
        // color_buffers[0].dtype().number_of_elements()

        // std::cout << "$$$ depth_buffers " << depth_buffers.size() << std::endl;
        // for (size_t i = 0; i < color_buffers[0].dtype().number_of_elements(); i++)
        // {
        //     unsigned char v = color_buffers[0].as_char_ptr()[i];
        //     if (v > 0)
        //         std::cout << v << " ";
        // }
        // std::cout << std::endl;

        ascent_probing.close();
        int probing_images = int(std::round(probing_factor * phi * theta));
        log_time(start, "probing " + std::to_string(probing_images) + " ", world_rank);
        // TODO: test: use total time measurement instead of image times
        render_times.clear();
        std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
        render_times.push_back(elapsed.count() * 1000.0);
    }
    else
    {
        render_times.push_back(100.f); // add dummy value for in transit test only
    }

#if ASCENT_MPI_ENABLED
    if (probing_factor < 1.0)
    {
        splitAndRender(mpi_comm_world, world_size, world_rank, sim_comm, rank_split, 
                       render_times, phi*theta, m_data, render_chunks, 
                       probing_factor, vis_budget);
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
