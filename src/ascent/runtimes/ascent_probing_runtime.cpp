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
#include <vtkm/filter/VectorMagnitude.h>

#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/rendering/Image.hpp>
#include <vtkh/rendering/ImageCompositor.hpp>
#include <vtkh/rendering/Compositor.hpp>

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


void debug_break()
{
    volatile int vi = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == vi)
        sleep(5);
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

/**
 * 
 */
template <typename T>
std::vector<int> sort_indices(const std::vector<T> &v)
{
    std::vector<int> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), 
              [&v](int i, int j) { return v[i] < v[j]; } 
             );
    return indices;
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


int recv_any_using_schema(Node &node, const int src, const int tag, const MPI_Comm comm)
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

Node pack_node(const Node &node)
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
    
    Node n_msg;
    n_msg.set_schema(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    n_msg["schema_len"].set((int64)snd_schema_json.length());
    n_msg["schema"].set(snd_schema_json);
    n_msg["data"].update(node);

    return n_msg;
}

void unpack_node(const Node &node, Node &unpacked)
{
    // debug_break();

    uint8 *n_buff_ptr = (uint8*)node.data_ptr();

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
    n_msg["data"].set_external(rcv_schema, n_buff_ptr);
    
    // copy out to our result node TODO: copy?
    unpacked.update(n_msg["data"]);  
}


int 
ibsend_using_schema(const Node &node, int dest, int tag, MPI_Comm comm, MPI_Request* request)
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

    int mpi_error = MPI_Ibsend(const_cast<void*>(n_msg.data_ptr()),
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


// int ibsend_using_schema(const Node &n_msg, const int dest, const int tag, 
//                         MPI_Comm comm, MPI_Request *request) 
// {     
//     index_t msg_data_size = n_msg.total_bytes_compact();

//     int mpi_error = MPI_Bsend(const_cast<void*>(n_msg.data_ptr()),
//                                static_cast<int>(msg_data_size),
//                                MPI_BYTE,
//                                dest,
//                                tag,
//                                comm);
//                             //    request);
    
//     if (mpi_error)
//         std::cout << "ERROR sending dataset to " << dest << std::endl;

//     return mpi_error;
// }

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
int calc_render_msg_size(const int render_count, const double probing_factor,
                         const int width = 1024, const int height = 1024, 
                         const int channels = 4+4)
{
    const int total_renders = render_count - int(probing_factor * render_count);
    const int overhead_render = 396 + 256;    // TODO: add bytes for name
    const int overhead_global = 288;
    return total_renders * channels * width * height + 
            total_renders * overhead_render + overhead_global;
}


struct RenderBatch
{
    int runs = 0;
    int rest = 0;
};

RenderBatch get_batch(const int render_count, const int batch_size)
{
    RenderBatch b;
    b.runs = int(std::ceil(render_count / double(batch_size)));
    // last run may have less renders than the other batches
    b.rest = render_count % batch_size;
    return b;
}

void post_irecv_renders(std::vector< std::vector< std::vector<int>>> &renders,
                        std::vector<MPI_Request> &requests,
                        std::vector<RenderBatch> &batches,
                        const vector<int> &src_ranks,
                        const int sending_count,
                        const int batch_size,
                        MPI_Comm comm,
                        int probing_count,
                        double probing_factor
                       )
{
    for (int i = 0; i < sending_count; ++i)
    {
        for (int j = 0; j < batches[i].runs; ++j)
        {
            // correct size for last iteration
            const int current_batch_size = (j == batches[i].runs - 1) ? batches[i].rest : batch_size;
            // std::cout << " ~~~ current_batch_size " << world_rank  << " batch size " << current_batch_size
            //         << std::endl;
            if (current_batch_size == 0)
                break;

            const int buffer_size = calc_render_msg_size(current_batch_size, probing_factor);

            // Node n_buffer(DataType::uint8(buffer_size));
            std::vector<int> buffer(buffer_size);
            renders[i].push_back(buffer);
            int mpi_error = MPI_Irecv(renders[i].back().data(),
                                        buffer_size,
                                        MPI_INT,
                                        src_ranks[i],
                                        j+1,
                                        comm,
                                        &requests[i]
                                        );
            if (mpi_error)
                std::cout << "ERROR receiving dataset from " << src_ranks[i] << std::endl;
        }
    }
}

/**
 * Make a unique pointer (for backward compatability, native since c++14).
 */
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) 
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
 * @return the number of probing renders in a given render sequence.
 */
int get_probing_count(const int render_count, const int stride, const int render_offset = 0)
{
    int probing_count = 0;
    for (int i = render_offset; i < render_offset + render_count; i++)
    {
        if (i % stride == 0)
            ++probing_count;
    }

    return probing_count;
}

int get_current_batch_size(const int batch_size, const RenderBatch batch, const int iteration)
{
    int current_batch_size = batch_size;
    if ((iteration == batch.runs - 1) && (batch.rest != 0))
        current_batch_size = batch.rest;
    return current_batch_size;
}

//-----------------------------------------------------------------------------
// TODO: refactor this method: split up into prep, sim and vis parts
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const int world_size,
                    const int world_rank,
                    const MPI_Comm vis_comm,
                    const int sim_node_count,
                    const std::vector<double> &my_probing_times,
                    const int max_render_count,
                    conduit::Node &data,
                    conduit::Node &render_chunks_probing,
                    const double probing_factor,
                    const double vis_budget = 0.1)
{
    assert(vis_budget >= 0.0 && vis_budget <= 1.0);
    // vis_budget of 1.0 => all in transit

    assert(sim_node_count > 0 && sim_node_count <= world_size);

    int is_intransit = 0;
    int is_rendering = 0;
    int is_sending = 0;
    bool is_vis_node = false;

    int my_src_rank = -1;
    int my_dest_rank = -1;

    const int vis_node_count = world_size - sim_node_count;
    float my_vis_estimate = 0.f;
    float my_avg_probing_time = 0.f;
    float my_sim_estimate = data["state/sim_time"].to_float();

    Node data_packed = pack_node(data);;
    int my_data_size = 0;
    
    std::cout << "~~~ " << my_sim_estimate << " sec sim time estimate " 
              << world_rank << std::endl;

    const int probing_stride = max_render_count / (probing_factor*max_render_count);
    const int probing_count = get_probing_count(max_render_count, probing_stride);
    // int(std::round(probing_factor*max_render_count));
    // render count without probing renders
    const int render_count = max_render_count - probing_count;

    // nodes with the highest rank are vis nodes
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

        my_data_size = data_packed.total_bytes_compact();
    }

#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(mpi_comm_world);
    auto start0 = std::chrono::system_clock::now(); // TODO: timer position

    // gather all simulation time estimates
    std::vector<float> g_sim_estimates(world_size, 0.f);
    MPI_Allgather(&my_sim_estimate, 1, MPI_FLOAT, 
                  g_sim_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);
    // gather all visualization time estimates
    std::vector<float> g_vis_estimates(world_size, 0.f);
    MPI_Allgather(&my_avg_probing_time, 1, MPI_FLOAT, 
                  g_vis_estimates.data(), 1, MPI_FLOAT, mpi_comm_world);
    // gather all data set sizes
    std::vector<int> g_data_sizes(world_size, 0);
    MPI_Allgather(&my_data_size, 1, MPI_INT, g_data_sizes.data(), 1, MPI_INT, mpi_comm_world);

    // sort the ranks accroding to sim+vis time estimate
    std::vector<int> rank_order = sort_ranks(g_sim_estimates, g_vis_estimates);
    // assign sim nodes to vis nodes
    std::vector<int> node_map = node_assignment(rank_order, g_vis_estimates, vis_node_count);

    // DEBUG: OUT
    if (world_rank == 0)
    {
        std::cout << "=== node_map ";
        for (auto &a : node_map)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    // distribute rendering load across sim and vis loads
    std::vector<int> g_render_counts = load_assignment(g_sim_estimates, g_vis_estimates,
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

    // TODO: make batch_size variable by config, 
    // TODO: adapt batch_size to probing size so that first render is always probing,
    //       this would avoid batch size 1 issues
    const int BATCH_SIZE = 20;  

    // renders setup
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int CHANNELS = 4 + 4; // RGBA + depth (float)
    
    // mpi message tags
    const int tag_data = 0;
    const int tag_probing = tag_data + 1;
    const int tag_inline = tag_probing + 1;

    // common options for both sim and vis nodes
    Node ascent_opts, blank_actions;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_comm_world);
    ascent_opts["actions_file"] = "cinema_actions.yaml";
    ascent_opts["is_probing"] = 0;
    ascent_opts["probing_factor"] = probing_factor;

    log_time(start0, "- load distribution ", world_rank);

    if (is_vis_node) // vis nodes 
    {
        // find all sim nodes sending data to this vis node
        std::vector<int> sending_node_ranks;
        for (int i = 0; i < sim_node_count; ++i)
        {
            if (node_map[i] == my_dest_rank)
                sending_node_ranks.push_back(i);
        }
        const int sending_count = int(sending_node_ranks.size());
        std::map<int, int> sending_counts;
        for (auto n : node_map)
            ++sending_counts[n];

        std::vector<int> depth_id_order;
        for (int i = 0; i < vis_node_count; i++)
        {
            std::vector<int>::iterator it = node_map.begin();
            while ((it = std::find_if(it, node_map.end(), [&](int x){return x == i; })) 
                    != node_map.end())
            {
                int d = std::distance(node_map.begin(), it);
                depth_id_order.push_back(d);
                it++;
            }
        }
        std::cout << "== depth_id_order ";
        for(auto a : depth_id_order)
            std::cout << a << " ";
        std::cout << std::endl;


        std::stringstream node_string;
        std::copy(sending_node_ranks.begin(), sending_node_ranks.end(), 
                    std::ostream_iterator<int>(node_string, " "));
        std::cout << "~~~~ vis node " << world_rank << ": receives extract(s) from " 
                  << node_string.str() << std::endl;

        const std::vector<int> src_ranks = sending_node_ranks;
        std::vector< std::unique_ptr<Node> > render_chunks_vis(sending_count);
        std::vector< std::unique_ptr<Node> > datasets(sending_count);

        // receive all data sets (blocking)
        // for (int i = 0; i < sending_count; ++i)
        // {
        //     auto start = std::chrono::system_clock::now();
        //     conduit::Node dataset;
        //     src_ranks[i] = recv_any_using_schema(dataset, MPI_ANY_SOURCE, tag_data, mpi_comm_world);
        //     datasets[i] = make_unique<Node>(dataset);
        //     std::cout << "~~~ vis node " << world_rank << " received data from "
        //               << src_ranks[i] << std::endl;
        //     log_time(start, "- receive data ", world_rank);
        // }

        // post recv for datasets
        std::vector<MPI_Request> requests_data(sending_count, MPI_REQUEST_NULL);
        for (int i = 0; i < sending_count; ++i)
        {
            datasets[i] = make_unique<Node>(DataType::uint8(g_data_sizes[src_ranks[i]])); // +4250

            int mpi_error = MPI_Irecv(datasets[i]->data_ptr(),
                                      datasets[i]->total_bytes_compact(),
                                      MPI_BYTE,
                                      src_ranks[i],
                                      tag_data,
                                      mpi_comm_world,
                                      &requests_data[i]
                                      );
            if (mpi_error)
                std::cout << "ERROR receiving dataset from " << src_ranks[i] << std::endl;

            // std::cout << "~~~ vis node " << world_rank << " receiving " << g_data_sizes[src_ranks[i]]
            //           << " bytes from " << src_ranks[i] << std::endl;
        }

        // every associated sim node sends n batches of renders to this vis node
        std::vector<RenderBatch> batches(sending_count);
        for (int i = 0; i < batches.size(); ++i)
        {
            int render_count = g_render_counts[src_ranks[i]] 
                                + int(g_render_counts[src_ranks[i]]*probing_factor);
            batches[i] = get_batch(render_count, BATCH_SIZE);
        }

        int sum_batches = 0;
        for (const auto &b : batches)
            sum_batches += b.runs;

        // probing chunks
        std::vector< std::unique_ptr<Node> > render_chunks_probe(sending_count);
        std::vector<MPI_Request> requests_probing(sending_count, MPI_REQUEST_NULL);
        // render chunks sim
        // senders / batches / renders
        std::vector< std::vector< std::unique_ptr<Node> > > render_chunks_sim(sending_count);
        std::vector< std::vector<MPI_Request>> requests_inline_sim(sending_count);

        // pre-allocate the mpi receive buffers
        for (int i = 0; i < sending_count; i++)
        {   
            int buffer_size = calc_render_msg_size(probing_count, 0.0);
            render_chunks_probe[i] = make_unique<Node>(DataType::uint8(buffer_size));

            render_chunks_sim[i].resize(batches[i].runs);
            requests_inline_sim[i].resize(batches[i].runs, MPI_REQUEST_NULL); // +1 ??

            for (int j = 0; j < batches[i].runs; ++j)
            {
                const int current_batch_size = get_current_batch_size(BATCH_SIZE, batches[i], j);
                buffer_size = calc_render_msg_size(current_batch_size, probing_factor);
                render_chunks_sim[i][j] = make_unique<Node>(DataType::uint8(buffer_size));
                if (current_batch_size == 1)    // TODO: single render that was already probed
                    render_chunks_sim[i].pop_back();
                // std::cout << current_batch_size << " expected render_msg_size " 
                //           << buffer_size << std::endl;
            }
        }

        // post the receives for the render chunks to receive asynchronous (non-blocking)
        for (int i = 0; i < sending_count; ++i)
        {
            std::cout << " ~~~ vis node " << world_rank << " receiving " << batches[i].runs
                      << " render chunks from " << src_ranks[i] << std::endl;

            // receive probing render chunks
            if (vis_budget < 1.0)   // 1 implies in transit only, i.e. we don't use probing
            {
                int mpi_error = MPI_Irecv(render_chunks_probe[i]->data_ptr(),
                                          render_chunks_probe[i]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          tag_probing,
                                          mpi_comm_world,
                                          &requests_probing[i]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving probing parts from " << src_ranks[i] << std::endl;
            }

            for (int j = 0; j < batches[i].runs; ++j)
            {
                // correct size for last iteration
                const int current_batch_size = get_current_batch_size(BATCH_SIZE, batches[i], j);
                if (current_batch_size <= 1) // TODO: single render that was already probed
                    break;

                int mpi_error = MPI_Irecv(render_chunks_sim[i][j]->data_ptr(),
                                          render_chunks_sim[i][j]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          tag_inline + j,
                                          mpi_comm_world,
                                          &requests_inline_sim[i][j]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving render parts from " << src_ranks[i] << std::endl;
            }
        }

        // wait for all data sets to arrive
        for (int i = 0; i < sending_count; ++i)
        {
            int id = -1;
            auto start1 = std::chrono::system_clock::now();
            MPI_Waitany(requests_data.size(), requests_data.data(), &id, MPI_STATUS_IGNORE);
            log_time(start1, "- receive data ", world_rank);
        }

        // render all data sets
        for (int i = 0; i < sending_count; ++i)
        {
            Node dataset;
            unpack_node(*datasets[i], dataset);

            Node verify_info;
            if (conduit::blueprint::mesh::verify(dataset, verify_info))
            {
                // vis node needs to render what is left
                const int current_render_count = max_render_count - g_render_counts[src_ranks[i]];
                const int render_offset = max_render_count - current_render_count;

                auto start = std::chrono::system_clock::now();
                if (current_render_count > 0)
                {
                    std::cout   << "~~~~ VIS node " << world_rank << " rendering " 
                                << render_offset << " - " 
                                << render_offset + current_render_count << std::endl;
                    ascent_opts["render_count"] = current_render_count;
                    ascent_opts["render_offset"] = render_offset;
                    // TODO: change variable name to cinema increment indicator
                    ascent_opts["vis_node"] = (i == 0) ? true : false;  

                    Ascent ascent_render;
                    ascent_render.open(ascent_opts);
                    ascent_render.publish(dataset);
                    ascent_render.execute(blank_actions);
                    
                    conduit::Node info;
                    ascent_render.info(info);
                    // NodeIterator itr = info["render_times"].children();
                    // while (itr.has_next())
                    // {
                    //     Node &t = itr.next();
                    //     // render_times.push_back(t.to_double());
                    //     std::cout << t.to_double() << " ";
                    // }
                    // std::cout << "render times VIS node +++ " << world_rank << std::endl;

                    // TODO: deep copy happening here? (if so directly use info node)
                    Node render_chunks;
                    render_chunks["depths"] = info["depths"];
                    render_chunks["color_buffers"] = info["color_buffers"];
                    render_chunks["depth_buffers"] = info["depth_buffers"];
                    render_chunks["render_file_names"] = info["render_file_names"];
                    render_chunks_vis[i] = make_unique<Node>(render_chunks);

                    ascent_render.close();
                }
                log_time(start, "+ render vis " + std::to_string(current_render_count) + " ", world_rank);
            }
            else
            {
                std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                            << std::endl;
            }
        }   // for: render all datasets sent

        {   // wait for receive of render chunks to complete
            auto t_start = std::chrono::system_clock::now();
            // renders from probing            
            MPI_Waitall(requests_probing.size(), requests_probing.data(), MPI_STATUSES_IGNORE);
            std::cout << "~~~~wait for receive in line " << world_rank << std::endl;
            // inline renders
            for (auto &batch_requests : requests_inline_sim)
                MPI_Waitall(batch_requests.size(), batch_requests.data(), MPI_STATUSES_IGNORE);
            log_time(t_start, "+ wait receive img ", world_rank);
        }

        {   // multi node compositing
            auto t_start = std::chrono::system_clock::now();

            // unpack sent renders -> NOTE: takes too long, can we avoid copies?
            std::vector<std::unique_ptr<Node> > parts_probing;
            // render parts from this vis node don't need to be unpacked
            for (auto const& p : render_chunks_probe)
            {
                parts_probing.emplace_back(make_unique<Node>());
                unpack_node(*p, *parts_probing.back());
            }
            // sender / batches
            std::vector<std::vector<std::unique_ptr<Node> > > parts_sim(sending_count);
            for (int i = 0; i < sending_count; ++i)
            {
                for (auto const& batch : render_chunks_sim[i])
                {
                    parts_sim[i].emplace_back(make_unique<Node>());
                    unpack_node(*batch, *parts_sim[i].back());
                }
            }
            log_time(t_start, "+ unpack images ", world_rank);
            std::cout << "~~~~finished unpacking images, compositing... " << world_rank << std::endl;

            // arrange render order
            t_start = std::chrono::system_clock::now();
            
            vector<int> probing_enum_sim(sending_count, 0);
            vector<int> probing_enum_vis(sending_count, 0);
            // images / sender / values
            std::vector<std::vector<float> > depths(max_render_count);
            std::vector<std::vector<std::string> > render_file_names(max_render_count);
            std::vector<std::vector<std::unique_ptr<Node> > > color_buffers(max_render_count);
            std::vector<std::vector<std::unique_ptr<Node> > > depth_buffers(max_render_count);
            for (int j = 0; j < max_render_count; ++j)
            {
                // std::cout << "\nimage " << j << std::endl;
                for (int i = 0; i < sending_count; ++i)
                {
                    // std::cout << "  " << i << " ";
                    depths[j].reserve(sending_count);
                    color_buffers[j].reserve(sending_count);
                    depth_buffers[j].reserve(sending_count);

                    if (j % probing_stride == 0)    // probing image
                    {
                        const index_t id = j / probing_stride;
                        // std::cout << " " << world_rank << " probe  " << id << std::endl;

                        depths[j].push_back((*parts_probing[i])["depths"].child(id).to_float());
                        render_file_names[j].push_back((*parts_probing[i])["render_file_names"].child(id).as_string());
                        std::unique_ptr<Node> cb = make_unique<Node>((*parts_probing[i])["color_buffers"].child(id));
                        color_buffers[j].push_back(std::move(cb));
                        std::unique_ptr<Node> db = make_unique<Node>((*parts_probing[i])["depth_buffers"].child(id));
                        depth_buffers[j].push_back(std::move(db));
                        // keep track of probing images
                        {
                            // reset counter if first in batch
                            if (j % BATCH_SIZE == 0)   
                                probing_enum_sim[i] = 0;
                            ++probing_enum_sim[i];

                            // reset probing counter if first render in vis chunks
                            if (j - g_render_counts[src_ranks[i]] == 0)
                                probing_enum_vis[i] = 0;
                            ++probing_enum_vis[i];
                        }
                    }
                    else if (j < g_render_counts[src_ranks[i]]) // part comes from sim node (inline)
                    {
                        // reset probing counter if first in batch and not a probing render
                        if (j % BATCH_SIZE == 0)
                            probing_enum_sim[i] = 0;

                        const int batch_id = j / BATCH_SIZE;
                        const index_t id = (j % BATCH_SIZE) - probing_enum_sim[i];
                        // std::cout << " " << world_rank << " sim  " << id << std::endl;

                        depths[j].push_back((*parts_sim[i][batch_id])["depths"].child(id).to_float());                        
                        render_file_names[j].push_back((*parts_sim[i][batch_id])["render_file_names"].child(id).as_string());
                        std::unique_ptr<Node> cb = make_unique<Node>((*parts_sim[i][batch_id])["color_buffers"].child(id));
                        color_buffers[j].push_back(std::move(cb));
                        std::unique_ptr<Node> db = make_unique<Node>((*parts_sim[i][batch_id])["depth_buffers"].child(id));
                        depth_buffers[j].push_back(std::move(db));
                    }
                    else    // part rendered on this vis node
                    {
                        // reset probing counter if first render in vis chunks and not a probing render
                        if (j - g_render_counts[src_ranks[i]] == 0)
                            probing_enum_vis[i] = 0;

                        const index_t id = (j - g_render_counts[src_ranks[i]]) - probing_enum_vis[i];
                        // std::cout << " " << world_rank << " vis  " << id << std::endl;

                        depths[j].push_back((*render_chunks_vis[i])["depths"].child(id).to_float());
                        render_file_names[j].push_back((*render_chunks_vis[i])["render_file_names"].child(id).as_string());
                        std::unique_ptr<Node> cb = make_unique<Node>((*render_chunks_vis[i])["color_buffers"].child(id));
                        color_buffers[j].push_back(std::move(cb));
                        std::unique_ptr<Node> db = make_unique<Node>((*render_chunks_vis[i])["depth_buffers"].child(id));
                        depth_buffers[j].push_back(std::move(db));
                    }
                }
            }
            log_time(t_start, "+ arrange renders ", world_rank);

            // actual compositing
            auto t_start0 = std::chrono::system_clock::now();
            
            // Set the vis_comm to be the vtkh comm.
            vtkh::SetMPICommHandle(int(MPI_Comm_c2f(vis_comm)));
            int vis_rank = 0;
            MPI_Comm_rank(vis_comm, &vis_rank);

            // DEBUG: 
            // debug_break();

            // Set the number of receiving depth values per node 
            // and the according displacements.
            std::vector<int> counts_recv;
            std::vector<int> displacements(1, 0); 
            for (const auto &e : sending_counts)
            {
                counts_recv.push_back(e.second);
                displacements.push_back(displacements.back() + e.second);
            }
            displacements.pop_back();

            // loop over images (camera positions)
            for (int j = 0; j < max_render_count; ++j)
            {
                t_start = std::chrono::system_clock::now();
                // gather all dephts values from vis nodes
                std::vector<float> v_depths(sim_node_count, 0.f);
                MPI_Allgatherv(depths[j].data(), depths[j].size(), MPI_FLOAT, 
                               v_depths.data(), counts_recv.data(), displacements.data(), 
                               MPI_FLOAT, vis_comm);

                std::vector<std::pair<float, int> > depth_id;
                for (int k = 0; k < v_depths.size(); k++)
                    depth_id.push_back(std::make_pair(v_depths[k], depth_id_order[k]));
                // sort based on depth values
                std::sort(depth_id.begin(), depth_id.end());

                // convert the depth order to an integer ranking
                std::vector<int> depths_order;
                for(auto it = depth_id.begin(); it != depth_id.end(); it++)
                    depths_order.push_back(it->second);
                
                // get a mapping from MPI src rank to depth rank
                std::vector<int> depths_order_id = sort_indices(depths_order);

                vtkh::Compositor compositor;
                compositor.SetCompositeMode(vtkh::Compositor::VIS_ORDER_BLEND);
                
                // loop over render parts (= 1 per sim node) and add as images
                for (int i = 0; i < sending_count; ++i)
                {
                    const int id = depths_order_id[src_ranks[i]];
                    compositor.AddImage(color_buffers[j][i]->as_unsigned_char_ptr(),
                                        depth_buffers[j][i]->as_float_ptr(),
                                        WIDTH,
                                        HEIGHT,
                                        id
                                        );
                }

                // composite
                vtkh::Image result = compositor.Composite();
                log_time(t_start, "+ compositing image ", world_rank);

                if (vis_rank == 0)
                {   // write to disk 
                    t_start = std::chrono::system_clock::now();
                    result.Save(render_file_names[j][0]);
                    log_time(t_start, "+ save image ", world_rank);
                }
            }
            log_time(t_start0, "+ compositing total ", world_rank);
        }

    } // end vis node
    else // SIM nodes
    {
        const int destination = node_map[world_rank] + sim_node_count;
        std::cout << "~~~~rank " << world_rank << ": sends extract to " 
                  <<  node_map[world_rank] + sim_node_count << std::endl;

        // std::vector<Node> compact_img_batches;
        Node verify_info;
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            Node compact_probing_renders = pack_node(render_chunks_probing);
            const int my_render_count = g_render_counts[world_rank] + int(g_render_counts[world_rank]*probing_factor);
            RenderBatch batch = get_batch(my_render_count, BATCH_SIZE);

            {   // init send buffer
                detach_mpi_buffer();

                const index_t msg_size_render = calc_render_msg_size(my_render_count, probing_factor);
                const index_t msg_size_probing = compact_probing_renders.total_bytes_compact();
                const int overhead = MPI_BSEND_OVERHEAD * (batch.runs + 2); // 1 probing batch
                MPI_Buffer_attach(malloc(msg_size_render + msg_size_probing + my_data_size + overhead), 
                                         msg_size_render + msg_size_probing + my_data_size + overhead);
                // std::cout << "-- buffer size: " << (msg_size + msg_size_probing + overhead) << std::endl;
            }

            {   // send data to vis node
                auto t_start = std::chrono::system_clock::now();
                int mpi_error = MPI_Ssend(data_packed.data_ptr(),
                                        data_packed.total_bytes_compact(),
                                        MPI_BYTE,
                                        destination,
                                        tag_data,
                                        mpi_comm_world
                                        );
                if (mpi_error)
                    std::cout << "ERROR sending sim data to " << destination << std::endl;
                log_time(t_start, "- send data ", world_rank);
            }

            
            MPI_Request request_probing = MPI_REQUEST_NULL;
            {   // send probing chunks to vis node
                int mpi_error = MPI_Ibsend(compact_probing_renders.data_ptr(),
                                           compact_probing_renders.total_bytes_compact(),
                                           MPI_BYTE,
                                           destination,
                                           tag_probing,
                                           mpi_comm_world,
                                           &request_probing
                                           );
                if (mpi_error)
                    std::cout << "ERROR sending probing renders to " << destination << std::endl;
            }

            // in line rendering using ascent
            Ascent ascent_render;
            std::vector<MPI_Request> requests(batch.runs, MPI_REQUEST_NULL);
            auto t_start = std::chrono::system_clock::now();
            for (int i = 0; i < batch.runs; ++i)
            {
                const int begin = i*BATCH_SIZE;
                const int current_batch_size = get_current_batch_size(BATCH_SIZE, batch, i);
                if (current_batch_size <= 1)
                    break;
                const int end = i*BATCH_SIZE + current_batch_size;

                std::cout   << "~~ SIM node " << world_rank << " rendering " 
                            << begin << " - " << end << std::endl;
                
                ascent_opts["render_count"] = end - begin;
                ascent_opts["render_offset"] = begin;
                ascent_opts["vis_node"] = false;

                ascent_render.open(ascent_opts);
                ascent_render.publish(data);
                ascent_render.execute(blank_actions);

                // send render chunks
                conduit::Node info;
                ascent_render.info(info);
                Node render_chunks_inline;
                render_chunks_inline["depths"] = info["depths"];
                render_chunks_inline["color_buffers"] = info["color_buffers"];
                render_chunks_inline["depth_buffers"] = info["depth_buffers"];
                render_chunks_inline["render_file_names"] = info["render_file_names"];

                // NodeIterator itr = info["color_buffers"].children();
                // while (itr.has_next())
                // {
                //     Node &b = itr.next();
                //     std::vector<unsigned char> render_raw;
                //     b.serialize(render_raw);
                //     cb_raw[i].reserve(cb_raw[i].size() + render_raw.size());
                //     std::move(render_raw.begin(), render_raw.end(), 
                //                 std::inserter(cb_raw[i], cb_raw[i].end()));
                    
                //     // for (size_t val = 0; val < b.dtype().number_of_elements(); val++)
                //     //     cb_raw[i].push_back(b.as_uint8_ptr()[val]);
                // }

                ascent_render.close();

                // send render chunks to vis node for compositing
                Node compact_renders = pack_node(render_chunks_inline);
                {
                    int mpi_error = MPI_Ibsend(compact_renders.data_ptr(),
                                               compact_renders.total_bytes_compact(),
                                               MPI_BYTE,
                                               destination,
                                               tag_inline + i,
                                               mpi_comm_world,
                                               &requests[i]
                                               );
                    if (mpi_error)
                        std::cout << "ERROR sending dataset to " << destination << std::endl;

                    // std::cout << world_rank << " end render sim " << current_batch_size
                    //             << " renders size " << compact_renders.total_bytes_compact() 
                    //             << std::endl;
                }                 
            }
            log_time(t_start, "+ render sim " + std::to_string(g_render_counts[world_rank]) + " ", world_rank);

            {   // wait for all sent data to be received
                t_start = std::chrono::system_clock::now();
                // probing
                MPI_Wait(&request_probing, MPI_STATUS_IGNORE);
                // render chunks
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                log_time(t_start, "+ wait send img ", world_rank);
            }
        }
        else
        {
            std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                        << std::endl;
        }
    } // end sim node

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


    int sim_count = 0;
#if ASCENT_MPI_ENABLED
    sim_count = int(std::round(world_size * node_split));
    int color = 0;
    if (world_rank >= sim_count)
        color = 1;
    const int vis_count = world_size - sim_count;

    // construct simulation comm
    std::vector<int> sim_ranks(sim_count);
    std::iota(sim_ranks.begin(), sim_ranks.end(), 0);
    std::vector<int> vis_ranks(vis_count);
    std::iota(vis_ranks.begin(), vis_ranks.end(), sim_count);

    MPI_Group world_group;
    MPI_Comm_group(mpi_comm_world, &world_group);

    MPI_Group sim_group;
    MPI_Group_incl(world_group, sim_count, sim_ranks.data(), &sim_group);
    MPI_Comm sim_comm;
    MPI_Comm_create_group(mpi_comm_world, sim_group, 0, &sim_comm);

    MPI_Group vis_group;
    MPI_Group_incl(world_group, vis_count, vis_ranks.data(), &vis_group);
    MPI_Comm vis_comm;
    MPI_Comm_create_group(mpi_comm_world, vis_group, 1, &vis_comm);

    ascent_opt["mpi_comm"] = MPI_Comm_c2f(sim_comm);
#endif // ASCENT_MPI_ENABLED

    std::vector<double> render_times;
    // TODO: handle corner case where there is no probing (and no probing chunks)
    Node render_chunks;
    // run probing only if this is a sim node
    if (world_rank < sim_count && probing_factor > 0.0)
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
        render_chunks["render_file_names"] = info["render_file_names"];

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
    if (probing_factor < 1.0) // probing_factor of 1 implies inline rendering only
    {
        splitAndRender(mpi_comm_world, world_size, world_rank, vis_comm, sim_count, 
                       render_times, phi*theta, m_data, render_chunks, 
                       probing_factor, vis_budget);
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&sim_group);
    MPI_Group_free(&vis_group);
    // MPI_Comm_free(&sim_comm); // Fatal error in PMPI_Comm_free: Invalid communicator
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
