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
#include <valarray>
#include <algorithm>
#include <ostream>
#include <iterator>

#include <thread>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

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


// void debug_break()
// {
//     volatile int vi = 0;
//     char hostname[256];
//     gethostname(hostname, sizeof(hostname));
//     printf("PID %d on %s ready for attach\n", getpid(), hostname);
//     fflush(stdout);
//     while (0 == vi)
//         sleep(5);
// }


// structs
//
//
struct MPI_Properties
{
    int size = 0;
    int rank = 0;
    int sim_node_count = 0;
    int vis_node_count = 0;
    MPI_Comm comm_world;
    MPI_Comm comm_vis;

    MPI_Properties(int size,
                   int rank,
                   int sim_node_count,
                   int vis_node_count,
                   MPI_Comm comm_world,
                   MPI_Comm comm_vis)
     : size(size)
     , rank(rank)
     , sim_node_count(sim_node_count)
     , vis_node_count(vis_node_count)
     , comm_world(comm_world)
     , comm_vis(comm_vis)
    {
    }
};

struct RenderConfig
{
    int max_count = 0;
    double probing_factor = 0.0;
    double vis_budget = 0.1;
    int probing_stride = 0;
    int probing_count = 0;
    int non_probing_count = 0;

    // TODO: make batch_size variable by config, 
    // TODO: Adapt batch_size to align with probing size so that first render is always probing,
    //       this would avoid batch size 1 issues.
    const static int BATCH_SIZE = 20;
    const static int WIDTH = 1024;
    const static int HEIGHT = 1024;
    const static int CHANNELS = 4 + 4; // RGBA + depth (float)

    RenderConfig(int max_render_count, double probing_factor = 0.0, double vis_budget = 0.1)
     : max_count(max_render_count)
     , probing_factor(probing_factor)
     , vis_budget(vis_budget)
    {
        // infer probing stride
        if (probing_factor <= 0.0)
            probing_stride = 0;
        else
            probing_stride = std::round(max_count / (probing_factor*max_count));

        // infer probing count
        probing_count = get_probing_count_part(max_count);

        // infer render count without probing renders
        non_probing_count = max_count - probing_count;
    }

    /**
     * Get the number of probings for a specific part of the rendering sequence.
     */
    int get_probing_count_part(const int render_count, const int render_offset = 0) const
    {
        int probing_count_part = 0;
        if (probing_stride <= 0)
            return probing_count_part;

        for (int i = render_offset; i < render_offset + render_count; i++)
        {
            if (i % probing_stride == 0)
                ++probing_count_part;
        }
        return probing_count_part;
    }
};

struct RenderBatch
{
    int runs = 0;
    int rest = 0;
};


/**
 * Assign part of the vis load to the vis nodes.
 */
std::vector<int> load_assignment(const std::vector<float> &sim_estimate, 
                                 const std::vector<float> &vis_estimates,
                                 const std::vector<int> &node_map,
                                 const RenderConfig render_cfg,
                                 const MPI_Properties mpi_props)
{
    // empirically determined render factor for sim nodes
    // TODO: investigate where this discrepancy comes from
    const float sim_factor = 1.2;       // 1.26 for d8, 1.1657 for n10, 1.2077 for n33
    // render factor for vis nodes
    const float vis_factor = 1.0;       // 0.9317 for n33, 1.0069 for n10

    assert(sim_estimate.size() == vis_estimates.size());
    
    std::valarray<float> t_inline(0.f, mpi_props.sim_node_count);
    for (size_t i = 0; i < mpi_props.sim_node_count; i++)
        t_inline[i] = vis_estimates[i] * sim_factor * render_cfg.non_probing_count;

    // TODO: add smart way to estimate compositing cost
    const float t_compositing = 0.2f * render_cfg.max_count;  // assume flat cost per image
    if (mpi_props.rank == 0)
        std::cout << "~~compositing estimate: " << t_compositing << std::endl;

    std::valarray<float> t_intransit(t_compositing, mpi_props.vis_node_count);
    std::valarray<float> t_sim(sim_estimate.data(), mpi_props.sim_node_count);

    std::vector<int> render_counts_sim(mpi_props.sim_node_count, 0);
    std::vector<int> render_counts_vis(mpi_props.vis_node_count, 0);

    // initially: push all vis load to vis nodes (=> all intransit case)
    for (size_t i = 0; i < mpi_props.sim_node_count; i++)
    {
        const int target_vis_node = node_map[i];

        t_intransit[target_vis_node] += t_inline[i] * (vis_factor/sim_factor);
        t_inline[i] = 0.f;
        render_counts_vis[target_vis_node] += render_cfg.non_probing_count;
    }

    // vis budget of 1 implies intransit only (i.e., only vis nodes render)
    if (render_cfg.vis_budget < 1.0)
    {
        // push back load to sim nodes until 
        // intransit time is smaller than max(inline + sim)
        // NOTE: this loop is ineffective w/ higher node counts
        int i = 0;
        std::valarray<float> t_inline_sim = t_inline + t_sim;
        while (t_inline_sim.max() < t_intransit.max()) 
        {
            // always push back to the fastest sim node
            const int min_id = std::min_element(begin(t_inline_sim), end(t_inline_sim)) 
                                - begin(t_inline_sim);

            // find the corresponding vis node 
            const int target_vis_node = node_map[min_id];

            if (render_counts_vis[target_vis_node] > 0)
            {
                t_intransit[target_vis_node] -= vis_estimates[min_id] * vis_factor;
                // Add render receive cost to vis node.
                // t_intransit[target_vis_node] += 0.09f;  
                render_counts_vis[target_vis_node]--;
            
                t_inline[min_id] += vis_estimates[min_id] * sim_factor;
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
            if (render_counts_sim[min_id] == render_cfg.non_probing_count)
                t_inline[min_id] = std::numeric_limits<float>::max() - t_sim[min_id];

            // recalculate inline + sim time
            t_inline_sim = t_inline + t_sim;
            ++i;
            if (i > render_cfg.non_probing_count * mpi_props.sim_node_count)
                ASCENT_ERROR("Error during load distribution.")
        }
    }

    std::vector<int> render_counts_combined(render_counts_sim);
    render_counts_combined.insert(render_counts_combined.end(), 
                                  render_counts_vis.begin(), 
                                  render_counts_vis.end());

    if (mpi_props.rank == 0)
    {
        std::cout << "=== render_counts ";
        for (auto &a : render_counts_combined)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    return render_counts_combined;
}


/**
 * Sort ranks in descending order according to sim + vis times estimations.
 */
std::vector<int> sort_ranks(const std::vector<float> &sim_estimates, 
                            const std::vector<float> &vis_estimates,
                            const int render_count)
{
    assert(sim_estimates.size() == vis_estimates.size());
    std::vector<int> rank_order(sim_estimates.size());
    std::iota(rank_order.begin(), rank_order.end(), 0);

    std::stable_sort(rank_order.begin(), 
                     rank_order.end(), 
                     [&](int i, int j) 
                     { 
                         return sim_estimates[i] + vis_estimates[i]*render_count 
                              > sim_estimates[j] + vis_estimates[j]*render_count;
                     } 
                     );
    return rank_order;
}

/**
 * Assign sim nodes to vis nodes based on their overall sim+vis times.
 */
std::vector<int> node_assignment(const std::vector<float> &g_sim_estimates, 
                                 const std::vector<float> &g_vis_estimates,
                                 const int vis_node_count, const int render_count)
{
    const std::vector<int> rank_order = sort_ranks(g_sim_estimates, g_vis_estimates, render_count);
    const int sim_node_count = rank_order.size() - vis_node_count;
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
        vis_node_cost[target_vis_node] += g_vis_estimates[rank_order[i]];
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
std::string get_timing_file_name(const int value, const int precision, const std::string &prefix,
                                 const std::string &path = "timings")
{
    std::ostringstream oss;
    oss << path;
    oss << "/";
    oss << prefix;
    oss << "_";
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
    std::ofstream out(get_timing_file_name(rank, 5, "vis"), std::ios_base::app);
    out << description << elapsed.count() << std::endl;
    out.close();
}

void log_global_time(const std::string &description,
                     const int rank)
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::ofstream out(get_timing_file_name(rank, 5, "global"), std::ios_base::app);
    out << description << " : " << millis << std::endl;
    out.close();
}

void print_time(std::chrono::time_point<std::chrono::system_clock> start, 
                const std::string &description,
                const int rank = -1)
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << description << elapsed.count() << " rank " << rank << std::endl;
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

void pack_node_external(Node &node, Node &packed)
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
    
    packed.set_schema(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    packed["schema_len"].set((int64)snd_schema_json.length());
    packed["schema"].set(snd_schema_json);
     
    // TODO: update_external is not working. 
    // Probably because setting the data external leads to non aligned memory?
    // -> Sending schema separate from data results in:
    // Fatal error in MPI_Ssend: Invalid buffer pointer, error stack:
    // MPI_Ssend(152): MPI_Ssend(buf=(nil)
    packed["data"].update_external(node);
}

Node pack_node(Node &node)
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

    // n_msg["schema"].print();

    // advance by the schema length
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    
    // apply the schema to the data
    n_msg["data"].set_external(rcv_schema, n_buff_ptr);
    
    // set data to our result node (external, no copy)
    unpacked.update_external(n_msg["data"]); 
    // unpacked.update(n_msg["data"]);
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

int get_current_batch_size(const int batch_size, const RenderBatch batch, const int iteration)
{
    int current_batch_size = batch_size;
    if ((iteration == batch.runs - 1) && (batch.rest != 0))
        current_batch_size = batch.rest;
    return current_batch_size;
}

typedef std::vector<std::shared_ptr<conduit::Node> > vec_node_sptr;
typedef std::vector<vec_node_sptr> vec_vec_node_sptr;

typedef std::vector<std::unique_ptr<conduit::Node> > vec_node_uptr;
typedef std::vector<vec_node_uptr> vec_vec_node_uptr;

static const int SLEEP = 0; // milliseconds

void save_image(vtkh::Image *image, const std::string &name)
{
    image->Save(name, true);
}

void image_consumer(std::mutex &mu, std::condition_variable &cond,
                    std::deque<std::pair<vtkh::Image *, std::string> > &buffer)
{
    std::cout << "Created consumer " << std::this_thread::get_id() << std::endl;
    while (true)
    {
        std::unique_lock<std::mutex> mlock(mu);
        cond.wait(mlock, [&buffer](){ return buffer.size() > 0; });
        std::pair<vtkh::Image *, std::string> image = buffer.front();
        // std::cout << "consumed " << image.second << std::endl;
        buffer.pop_front();
        mlock.unlock();
        cond.notify_all();

        if (image.first == nullptr && image.second == "KILL") // poison indicator 
        {
            std::cout << "Killed consumer " << std::this_thread::get_id() << std::endl;
            return;
        }
        else
        {
            image.first->Save(image.second, true);
        }
    }
}


/**
 *  Composite render chunks from probing, simulation nodes, and visualization nodes.
 */
void hybrid_compositing(const vec_node_uptr &render_chunks_probe, vec_vec_node_uptr &render_chunks_sim, 
                        const vec_node_sptr &render_chunks_vis, 
                        const std::vector<int> &g_render_counts, const std::vector<int> &src_ranks,
                        const std::vector<int> &depth_id_order, const std::map<int, int> &recv_counts,
                        const int my_vis_rank, 
                        const int my_render_recv_cnt, const int my_recv_cnt,
                        const RenderConfig &render_cfg, const MPI_Properties &mpi_props)
{
    auto t_start0 = std::chrono::system_clock::now();

    // unpack sent renders
    vec_node_sptr parts_probing;
    for (auto const& p : render_chunks_probe)
    {
        parts_probing.emplace_back(make_shared<Node>());
        unpack_node(*p, *parts_probing.back());
    }
    // sender / batches
    vec_vec_node_sptr parts_sim(my_render_recv_cnt);
    for (int i = 0; i < my_render_recv_cnt; ++i)
    {
        for (auto const& batch : render_chunks_sim[i])
        {
            parts_sim[i].emplace_back(make_shared<Node>());
            unpack_node(*batch, *parts_sim[i].back());
        }
    }
    
    std::cout << "~~~~arrange render order " << mpi_props.rank << std::endl;

    // arrange render order   
    vector<int> probing_enum_sim(my_render_recv_cnt, 0);
    vector<int> probing_enum_vis(my_recv_cnt, 0);
    // images / sender / values
    vec_vec_node_sptr render_ptrs(render_cfg.max_count);
    std::vector<std::vector<int> > render_arrangement(render_cfg.max_count);

    for (int j = 0; j < render_cfg.max_count; ++j)
    {
        render_ptrs[j].reserve(my_recv_cnt);
        render_arrangement[j].reserve(my_recv_cnt);

        // std::cout << "\nimage " << j << std::endl;
        for (int i = 0; i < my_recv_cnt; ++i)
        {
            // std::cout << "  " << i << " ";
            if ((render_cfg.probing_stride > 0) && (j % render_cfg.probing_stride == 0))    // probing image
            {
                const index_t id = j / render_cfg.probing_stride;
                // std::cout << " " << mpi_props.rank << " probe  " << id << std::endl;
                render_ptrs[j].emplace_back(parts_probing[i]);
                render_arrangement[j].emplace_back(id);

                {   // keep track of probing images
                    // reset counter if first in batch
                    if (j % render_cfg.BATCH_SIZE == 0)   
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
                // Reset the probing counter if this is the first render in a batch 
                // and this is not a probing render.
                if (j % render_cfg.BATCH_SIZE == 0)
                    probing_enum_sim[i] = 0;

                const int batch_id = j / render_cfg.BATCH_SIZE;
                const index_t id = (j % render_cfg.BATCH_SIZE) - probing_enum_sim[i];
                // std::cout << " " << mpi_props.rank << " sim  " << id << std::endl;
                render_ptrs[j].emplace_back(parts_sim[i][batch_id]);
                render_arrangement[j].emplace_back(id);
            }
            else    // part rendered on this vis node
            {
                // Reset the probing counter if this is the first render in vis node chunks 
                // and this is not a probing render.
                if (j - g_render_counts[src_ranks[i]] == 0)
                    probing_enum_vis[i] = 0;

                const index_t id = (j - g_render_counts[src_ranks[i]]) - probing_enum_vis[i];
                // std::cout << " " << mpi_props.rank << " vis  " << id << std::endl;
                render_ptrs[j].emplace_back(render_chunks_vis[i]);
                render_arrangement[j].emplace_back(id);
            }
        }
    }

    std::cout << "~~~~start compositing " << mpi_props.rank << std::endl;

    // Set the vis_comm to be the vtkh comm.
    vtkh::SetMPICommHandle(int(MPI_Comm_c2f(mpi_props.comm_vis)));

    // Set the number of receiving depth values per node 
    // and the according displacements.
    std::vector<int> counts_recv;
    std::vector<int> displacements(1, 0); 
    for (const auto &e : recv_counts)
    {
        counts_recv.push_back(e.second);
        displacements.push_back(displacements.back() + e.second);
    }
    displacements.pop_back();

    unsigned int thread_count = std::thread::hardware_concurrency();
    thread_count = std::min(thread_count, 16u);     // limit to 16 consumers to avoid overhead
    std::mutex mu;
    const int max_buffer_size = thread_count * 4;   // buffer a max of 4 images per thread
    std::condition_variable cond;
    std::deque<std::pair<vtkh::Image *, std::string> > buffer;
    std::vector<std::thread> consumers(thread_count);
    if (my_vis_rank == 0)
    {
        for (int i = 0; i < consumers.size(); ++i)
            consumers[i] = std::thread(&image_consumer, std::ref(mu), std::ref(cond), std::ref(buffer));
    }

    std::vector<vtkh::Image *> results(render_cfg.max_count);
    std::vector<std::thread> threads;
    // loop over images (camera positions)
// #pragma omp parallel for
    for (int j = 0; j < render_cfg.max_count; ++j)
    {
        // std::cout << "Compositing with thread count " << omp_get_num_threads() << std::endl;
        auto t_start = std::chrono::system_clock::now();
        // gather all dephts values from vis nodes
        std::vector<float> v_depths(mpi_props.sim_node_count, 0.f);

        std::vector<float> depths(my_recv_cnt);
        for (int i = 0; i < my_recv_cnt; i++)
            depths[i] = (*render_ptrs[j][i])["depths"].child(render_arrangement[j][i]).to_float();

        MPI_Allgatherv(depths.data(), depths.size(), 
                        MPI_FLOAT, v_depths.data(), counts_recv.data(), displacements.data(), 
                        MPI_FLOAT, mpi_props.comm_vis);

        std::vector<std::pair<float, int> > depth_id(v_depths.size());
        for (int k = 0; k < v_depths.size(); k++)
            depth_id[k] = std::make_pair(v_depths[k], depth_id_order[k]);
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
        for (int i = 0; i < my_recv_cnt; ++i)
        {
            const int id = depths_order_id[src_ranks[i]];
            compositor.AddImage((*render_ptrs[j][i])["color_buffers"].child(render_arrangement[j][i]).as_unsigned_char_ptr(),
                                (*render_ptrs[j][i])["depth_buffers"].child(render_arrangement[j][i]).as_float_ptr(),
                                render_cfg.WIDTH, render_cfg.HEIGHT, id);
        }

        // composite
        results[j] = compositor.CompositeNoCopy();

        // TODO: add screen annotations for hybrid (see vtk-h Scene::Render)
        // See vtk-h Renderer::ImageToCanvas() for how to get from result image to canvas.
        // Problem: we still need camera, ranges and color table.

        log_time(t_start, "+ compositing image ", mpi_props.rank);
        log_global_time("end composit image", mpi_props.rank);

        // save render using separate thread to hide latency
        if (my_vis_rank == 0)
        {
            std::string name = (*render_ptrs[j][0])["render_file_names"].child(render_arrangement[j][0]).as_string();

            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [&buffer, &max_buffer_size](){ return buffer.size() < max_buffer_size; });
            buffer.push_back(std::make_pair(results[j], name));
            // std::cout << "produced " << name << std::endl;
            locker.unlock();
            cond.notify_all();

            if (j == render_cfg.max_count - 1)  // last image -> kill consumers
            {  
                // poison consumers for cleanup
                for (int i = 0; i < consumers.size(); ++i)
                {
                    std::unique_lock<std::mutex> locker(mu);
                    cond.wait(locker, [&buffer, &max_buffer_size](){ return buffer.size() < max_buffer_size; });
                    buffer.push_back(std::make_pair(nullptr, std::string("KILL")));
                    locker.unlock();
                    cond.notify_all();
                }
                for (auto& t : consumers)
                    t.join();
            }
        }
    }
    log_time(t_start0, "+ compositing total ", mpi_props.rank);
    log_global_time("end compositing", mpi_props.rank);
}

void pack_and_send(Node& data, const int destination, const int tag, const MPI_Comm comm, MPI_Request &req)
{
    Node compact_node = pack_node(data);

    int mpi_error = MPI_Ibsend(compact_node.data_ptr(),
                               compact_node.total_bytes_compact(),
                               MPI_BYTE,
                               destination,
                               tag,
                               comm,
                               &req
                               );
    if (mpi_error)
        std::cout << "ERROR sending node to " << destination << std::endl;
}


//-----------------------------------------------------------------------------
void hybrid_render(const MPI_Properties &mpi_props,
                   const RenderConfig &render_cfg,
                   const std::vector<double> &my_probing_times,
                   const double total_probing_time,
                   conduit::Node &data,
                   conduit::Node &render_chunks_probing)
{
    auto start0 = std::chrono::system_clock::now();
    // render_cfg.vis_budget of 1.0 => all in transit
    assert(render_cfg.vis_budget >= 0.0 && render_cfg.vis_budget <= 1.0);
    assert(mpi_props.sim_node_count > 0 && mpi_props.sim_node_count <= mpi_props.size);

    bool is_vis_node = false;
    int my_vis_rank = -1;

    float my_avg_probing_time = 0.f;
    float my_sim_estimate = data["state/sim_time"].to_float();
    int my_data_size = 0;

    // TODO: no copy (set external) -> send schema separate from data?
    Node data_packed = pack_node(data);
    // Node data_packed;
    // pack_node_external(data, data_packed);

    std::cout << "~~~ " << my_sim_estimate << " sec sim time estimate " 
              << mpi_props.rank << std::endl;

    // nodes with the highest rank are vis nodes
    if (mpi_props.rank >= mpi_props.sim_node_count) 
    {
        is_vis_node = true;
        my_vis_rank = mpi_props.rank - mpi_props.sim_node_count;
        my_sim_estimate = 0.f;
    }
    else if (mpi_props.size > 1) // otherwise we are a sim node
    {
        assert(my_probing_times.size() > 0);
        
        bool use_time_per_render = false;
        if (use_time_per_render) // render time per probing image (w/o overhead)
        {
            double sum_render_times = std::accumulate(my_probing_times.begin(), 
                                                      my_probing_times.end(), 0.0);

            my_avg_probing_time = float(sum_render_times / my_probing_times.size());

            std::cout << "+++ probing times ";
            for (auto &a : my_probing_times)
                std::cout << a << " ";
            std::cout << mpi_props.rank << std::endl;
            std::cout << "probing w/o overhead " << sum_render_times/1000.0 << std::endl;
            std::cout << "probing w/  overhead " << total_probing_time << std::endl;
        }
        else // use whole probing time with overhead
        {
            my_avg_probing_time = total_probing_time / render_cfg.probing_count;
        }

        std::cout << "~~~ " << my_avg_probing_time 
                  << " sec vis time estimate per render " 
                  << mpi_props.rank << std::endl;

        my_data_size = data_packed.total_bytes_compact();
    }

#ifdef ASCENT_MPI_ENABLED
    // MPI_Barrier(mpi_props.comm_world);
    auto start1 = std::chrono::system_clock::now();

    // gather all simulation time estimates
    std::vector<float> g_sim_estimates(mpi_props.size, 0.f);
    MPI_Allgather(&my_sim_estimate, 1, MPI_FLOAT, 
                  g_sim_estimates.data(), 1, MPI_FLOAT, mpi_props.comm_world);
    // gather all visualization time estimates
    std::vector<float> g_vis_estimates(mpi_props.size, 0.f);
    MPI_Allgather(&my_avg_probing_time, 1, MPI_FLOAT, 
                  g_vis_estimates.data(), 1, MPI_FLOAT, mpi_props.comm_world);

    // NOTE: use maximum sim time for all nodes
    const float max_sim_time = *std::max_element(g_sim_estimates.begin(), g_sim_estimates.end());
    g_sim_estimates = std::vector<float>(mpi_props.size, max_sim_time);

    // assign sim nodes to vis nodes
    std::vector<int> node_map = node_assignment(g_sim_estimates, g_vis_estimates, 
                                                mpi_props.vis_node_count, 
                                                render_cfg.non_probing_count);

    // DEBUG: OUT
    if (mpi_props.rank == 0)
    {
        std::cout << "=== node_map ";
        for (auto &a : node_map)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    // distribute rendering load across sim and vis loads
    const std::vector<int> g_render_counts = load_assignment(g_sim_estimates, g_vis_estimates,
                                                             node_map, render_cfg, mpi_props);

    // gather all data set sizes for async recv
    std::vector<int> g_data_sizes(mpi_props.size, 0);
    MPI_Allgather(&my_data_size, 1, MPI_INT, g_data_sizes.data(), 1, MPI_INT, mpi_props.comm_world);
    
    // mpi message tags
    const int tag_data = 0;
    const int tag_probing = tag_data + 1;
    const int tag_inline = tag_probing + 1;

    // common options for both sim and vis nodes
    Node ascent_opts, blank_actions;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_props.comm_world);
    ascent_opts["actions_file"] = "cinema_actions.yaml";
    ascent_opts["is_probing"] = 0;
    ascent_opts["probing_factor"] = render_cfg.probing_factor;
    ascent_opts["insitu_type"] = (render_cfg.vis_budget >= 1.0) ? "intransit" : "hybrid";

    log_time(start1, "- load distribution ", mpi_props.rank);
    log_global_time("end loadAssignment", mpi_props.rank);

    if (is_vis_node) // vis nodes 
    {
        // find all sim nodes sending data to this vis node
        std::vector<int> sending_node_ranks;
        for (int i = 0; i < mpi_props.sim_node_count; ++i)
        {
            if (node_map[i] == my_vis_rank)
                sending_node_ranks.push_back(i);
        }
        const int my_recv_cnt = int(sending_node_ranks.size());
        // count of nodes that do inline rendering (0 for in transit case)
        const int my_render_recv_cnt = render_cfg.vis_budget >= 1.0 ? 0 : my_recv_cnt; 
        std::map<int, int> recv_counts;
        for (const auto &n : node_map)
            ++recv_counts[n];

        std::vector<int> depth_id_order;
        for (int i = 0; i < mpi_props.vis_node_count; i++)
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
        std::cout << "~~~~ vis node " << mpi_props.rank << ": receives extract(s) from " 
                  << node_string.str() << std::endl;

        const std::vector<int> src_ranks = sending_node_ranks;
        vec_node_sptr render_chunks_vis(my_recv_cnt, nullptr);
        std::vector<std::unique_ptr<Node> > datasets(my_recv_cnt);

        // post recv for datasets
        std::vector<MPI_Request> requests_data(my_recv_cnt, MPI_REQUEST_NULL);
        for (int i = 0; i < my_recv_cnt; ++i)
        {
            datasets[i] = make_unique<Node>(DataType::uint8(g_data_sizes[src_ranks[i]]));

            int mpi_error = MPI_Irecv(datasets[i]->data_ptr(),
                                      datasets[i]->total_bytes_compact(),
                                      MPI_BYTE,
                                      src_ranks[i],
                                      tag_data,
                                      mpi_props.comm_world,
                                      &requests_data[i]
                                      );
            if (mpi_error)
                std::cout << "ERROR receiving dataset from " << src_ranks[i] << std::endl;

            // std::cout << "~~~ vis node " << mpi_props.rank << " receiving " << g_data_sizes[src_ranks[i]]
            //           << " bytes from " << src_ranks[i] << std::endl;
        }

        // every associated sim node sends n batches of renders to this vis node
        std::vector<RenderBatch> batches(my_render_recv_cnt);
        for (int i = 0; i < batches.size(); ++i)
        {
            int render_count = g_render_counts[src_ranks[i]] 
                                + int(g_render_counts[src_ranks[i]]*render_cfg.probing_factor);
            batches[i] = get_batch(render_count, render_cfg.BATCH_SIZE);
        }

        // probing chunks
        vec_node_uptr render_chunks_probe(my_render_recv_cnt);
        std::vector<MPI_Request> requests_probing(my_render_recv_cnt, MPI_REQUEST_NULL);
        // render chunks sim
        // senders / batches / renders
        vec_vec_node_uptr render_chunks_sim(my_render_recv_cnt);
        std::vector< std::vector<MPI_Request>> requests_inline_sim(my_render_recv_cnt);

        // pre-allocate the mpi receive buffers
        for (int i = 0; i < my_render_recv_cnt; i++)
        {   
            int buffer_size = calc_render_msg_size(render_cfg.probing_count, 0.0);
            render_chunks_probe[i] = make_unique<Node>(DataType::uint8(buffer_size));

            render_chunks_sim[i].resize(batches[i].runs);
            requests_inline_sim[i].resize(batches[i].runs, MPI_REQUEST_NULL);

            for (int j = 0; j < batches[i].runs; ++j)
            {
                const int current_batch_size = get_current_batch_size(render_cfg.BATCH_SIZE, batches[i], j);
                buffer_size = calc_render_msg_size(current_batch_size, render_cfg.probing_factor);
                render_chunks_sim[i][j] = make_unique<Node>(DataType::uint8(buffer_size));
                if (current_batch_size == 1)    // TODO: single render that was already probed
                    render_chunks_sim[i].pop_back();
                // std::cout << current_batch_size << " expected render_msg_size " 
                //           << buffer_size << std::endl;
            }
        }

        // post the receives for the render chunks to receive asynchronous (non-blocking)
        for (int i = 0; i < my_render_recv_cnt; ++i)
        {
            // std::cout << " ~~~ vis node " << mpi_props.rank << " receiving " << batches[i].runs
            //           << " render chunks from " << src_ranks[i] << std::endl;
            // receive probing render chunks
            if (render_cfg.vis_budget < 1.0)   // 1 implies in transit only, i.e. we don't use probing
            {
                int mpi_error = MPI_Irecv(render_chunks_probe[i]->data_ptr(),
                                          render_chunks_probe[i]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          tag_probing,
                                          mpi_props.comm_world,
                                          &requests_probing[i]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving probing parts from " << src_ranks[i] << std::endl;
            }

            for (int j = 0; j < batches[i].runs; ++j)
            {
                // correct size for last iteration
                const int current_batch_size = get_current_batch_size(render_cfg.BATCH_SIZE, batches[i], j);
                if (current_batch_size <= 1) // TODO: single render that was already probed
                    break;

                int mpi_error = MPI_Irecv(render_chunks_sim[i][j]->data_ptr(),
                                          render_chunks_sim[i][j]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          tag_inline + j,
                                          mpi_props.comm_world,
                                          &requests_inline_sim[i][j]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving render parts from " << src_ranks[i] << std::endl;
            }
        }

        // wait for all data sets to arrive
        for (int i = 0; i < my_recv_cnt; ++i)
        {
            int id = -1;
            auto start1 = std::chrono::system_clock::now();
            MPI_Waitany(requests_data.size(), requests_data.data(), &id, MPI_STATUS_IGNORE);
            log_time(start1, "- receive data ", mpi_props.rank);
        }
        log_global_time("end receiveData", mpi_props.rank);

        // render all data sets
        std::vector<Ascent> ascent_renders(my_recv_cnt);
        for (int i = 0; i < my_recv_cnt; ++i)
        {
            // std::cout << "=== dataset size " << mpi_props.rank << " "
            //           << datasets[i]->total_bytes_compact() << std::endl;
            Node dataset;
            unpack_node(*datasets[i], dataset);

            Node verify_info;
            if (conduit::blueprint::mesh::verify(dataset, verify_info))
            {
                // vis node needs to render what is left
                const int current_render_count = render_cfg.max_count - g_render_counts[src_ranks[i]];
                const int render_offset = render_cfg.max_count - current_render_count;
                const int probing_count_part = render_cfg.get_probing_count_part(current_render_count, render_offset);

                auto start = std::chrono::system_clock::now();
                if (current_render_count > 0)
                {
                    std::cout   << "~~~~ VIS node " << mpi_props.rank << " rendering " 
                                << render_offset << " - " 
                                << render_offset + current_render_count << std::endl;
                    ascent_opts["render_count"] = current_render_count;
                    ascent_opts["render_offset"] = render_offset;
                    ascent_opts["cinema_increment"] = (i == 0) ? true : false;
                    ascent_opts["sleep"] = (src_ranks[i] == 0) ? SLEEP : 0;

                    ascent_renders[i].open(ascent_opts);
                    ascent_renders[i].publish(dataset);
                    ascent_renders[i].execute(blank_actions);

                    render_chunks_vis[i] = std::make_shared<Node>();
                    // ascent_main_runtime : out.set_external(m_info);
                    ascent_renders[i].info(*render_chunks_vis[i]);
                }
                log_time(start, "+ render vis " + std::to_string(current_render_count - probing_count_part) + " ", mpi_props.rank);
            }
            else
            {
                std::cout << "~~~~rank " << mpi_props.rank << ": could not verify sent data." 
                            << std::endl;
            }
        }   // for: render all datasets sent
        log_global_time("end render", mpi_props.rank);

        {   // wait for receive of render chunks to complete
            auto t_start = std::chrono::system_clock::now();
            // renders from probing            
            MPI_Waitall(requests_probing.size(), requests_probing.data(), MPI_STATUSES_IGNORE);
            std::cout << "~~~~wait for receive in line " << mpi_props.rank << std::endl;
            // inline renders
            for (auto &batch_requests : requests_inline_sim)
                MPI_Waitall(batch_requests.size(), batch_requests.data(), MPI_STATUSES_IGNORE);
            log_time(t_start, "+ wait receive img ", mpi_props.rank);
        }
        log_global_time("end receiveRenders", mpi_props.rank);

        hybrid_compositing(render_chunks_probe, render_chunks_sim, render_chunks_vis, 
                           g_render_counts, src_ranks, depth_id_order, recv_counts, my_vis_rank,
                           my_render_recv_cnt, my_recv_cnt, render_cfg, mpi_props);

        // Keep the vis node ascent instances open until render chunks have been processed.
        for (int i = 0; i < my_recv_cnt; i++)
            ascent_renders[i].close();  

    } // end vis node
    else // SIM nodes
    {
        const int destination = node_map[mpi_props.rank] + mpi_props.sim_node_count;
        // std::cout << "~~~~rank " << mpi_props.rank << ": sends extract to " 
        //           <<  node_map[mpi_props.rank] + mpi_props.sim_node_count << std::endl;
        Node verify_info;
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            // compact_probing_renders are now packed and send in separate thread
            // Node compact_probing_renders = pack_node(render_chunks_probing);
            const int my_render_count = g_render_counts[mpi_props.rank] + int(g_render_counts[mpi_props.rank]*render_cfg.probing_factor);
            RenderBatch batch = get_batch(my_render_count, render_cfg.BATCH_SIZE);

            {   // init send buffer
                detach_mpi_buffer();

                const index_t msg_size_render = calc_render_msg_size(my_render_count, render_cfg.probing_factor);
                const index_t msg_size_probing = calc_render_msg_size(render_cfg.probing_count, 0.0);
                // const index_t msg_size_probing = compact_probing_renders.total_bytes_compact();
                const int overhead = MPI_BSEND_OVERHEAD * (batch.runs + 2); // 1 probing batch
                MPI_Buffer_attach(malloc(msg_size_render + msg_size_probing + my_data_size + overhead), 
                                         msg_size_render + msg_size_probing + my_data_size + overhead);
                // std::cout << "-- buffer size: " << (msg_size + msg_size_probing + overhead) << std::endl;
            }

            {   // send data to vis node
                auto t_start = std::chrono::system_clock::now();
                int mpi_error = MPI_Ssend(const_cast<void*>(data_packed.data_ptr()),
                                          data_packed.total_bytes_compact(),
                                          MPI_BYTE,
                                          destination,
                                          tag_data,
                                          mpi_props.comm_world
                                          );
                if (mpi_error)
                    std::cout << "ERROR sending sim data to " << destination << std::endl;
                log_time(t_start, "- send data ", mpi_props.rank);
            }
            
            MPI_Request request_probing = MPI_REQUEST_NULL;
            // pack and send probing in own thread
            std::thread pack_renders_thread(&pack_and_send, std::ref(render_chunks_probing), 
                                            destination, tag_probing, mpi_props.comm_world, 
                                            std::ref(request_probing));

            log_global_time("end sendData", mpi_props.rank);

            // in line rendering using ascent
            std::vector<Ascent> ascent_renders(batch.runs);
            std::vector<MPI_Request> requests(batch.runs, MPI_REQUEST_NULL);
            std::vector<conduit::Node> info(batch.runs);
            Node render_chunks_inline;

            std::vector<std::thread> threads;
            // std::thread thread_pack(&test_function);
            auto t_start = std::chrono::system_clock::now();
            for (int i = 0; i < batch.runs; ++i)
            {
                const int begin = i*render_cfg.BATCH_SIZE;
                const int current_batch_size = get_current_batch_size(render_cfg.BATCH_SIZE, batch, i);
                if (current_batch_size <= 1)
                    break;
                const int end = i*render_cfg.BATCH_SIZE + current_batch_size;

                std::cout   << "~~ SIM node " << mpi_props.rank << " rendering " 
                            << begin << " - " << end << std::endl;
                
                ascent_opts["render_count"] = end - begin;
                ascent_opts["render_offset"] = begin;
                ascent_opts["insitu_type"] = "hybrid";
                ascent_opts["sleep"] = (mpi_props.rank == 0) ? SLEEP : 0;

                ascent_renders[i].open(ascent_opts);
                ascent_renders[i].publish(data);
                ascent_renders[i].execute(blank_actions);

                if (threads.size() > 0)
                {
                    threads.back().join();
                    threads.pop_back();
                }

                // send render chunks
                ascent_renders[i].info(info[i]);
                render_chunks_inline["depths"].set_external(info[i]["depths"]);
                render_chunks_inline["color_buffers"].set_external(info[i]["color_buffers"]);
                render_chunks_inline["depth_buffers"].set_external(info[i]["depth_buffers"]);
                render_chunks_inline["render_file_names"].set_external(info[i]["render_file_names"]);

                threads.push_back(std::thread(&pack_and_send, std::ref(render_chunks_inline), 
                                              destination, tag_inline + i, mpi_props.comm_world, 
                                              std::ref(requests[i])));
            }
            while (threads.size() > 0)
            {
                threads.back().join();
                threads.pop_back();
            }
            log_time(t_start, "+ render sim " + std::to_string(g_render_counts[mpi_props.rank]) + " ", mpi_props.rank);
            log_global_time("end render", mpi_props.rank);

            {   // wait for all sent data to be received
                pack_renders_thread.join();
                t_start = std::chrono::system_clock::now();
                // probing
                MPI_Wait(&request_probing, MPI_STATUS_IGNORE);
                // render chunks
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                log_time(t_start, "+ wait send img ", mpi_props.rank);
            }
            log_global_time("end sendRenders", mpi_props.rank);

            // Keep sim node ascent instances open until image chunks are sent.
            for (int i = 0; i < batch.runs; i++)
                ascent_renders[i].close();
        }
        else
        {
            std::cout << "~~~~rank " << mpi_props.rank << ": could not verify sent data." 
                        << std::endl;
        }
    } // end sim node

    log_time(start0, "___splitAndRun ", mpi_props.rank);
    // log_global_time("end hybridRender", mpi_props.rank);
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

    log_global_time("start probing", world_rank);
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
    bool is_inline = false;
    if (probing_factor >= 1.0) // probing_factor of 1 implies inline rendering only
        is_inline = true;

#if ASCENT_MPI_ENABLED
    sim_count = int(std::round(world_size * node_split));
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

    // only sim nodes have valid data
    ascent_opt["mpi_comm"] = MPI_Comm_c2f(sim_comm);
#endif // ASCENT_MPI_ENABLED

    std::vector<double> render_times;
    double total_probing_time = 0.0;
    // TODO: handle corner case where there is no probing (and no probing chunks)
    Ascent ascent_probing;
    Node render_chunks;
    // run probing only if this is a sim node
    if ((world_rank < sim_count && probing_factor > 0.0) || (probing_factor >= 1.0))
    {
        auto start = std::chrono::system_clock::now();
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime
        ascent_opt["is_probing"] = 1;
        ascent_opt["probing_factor"] = probing_factor;
        ascent_opt["render_count"] = phi * theta;
        ascent_opt["render_offset"] = 0;
        ascent_opt["insitu_type"] = is_inline ? "inline" : "hybrid";
        ascent_opt["sleep"] = world_rank == 0 ? SLEEP : 0;

        // all sim nodes run probing in a new ascent instance
        ascent_probing.open(ascent_opt);
        ascent_probing.publish(m_data);        // pass on data pointer
        ascent_probing.execute(probe_actions); // pass on actions

        if (!is_inline)
        {
            conduit::Node info;
            ascent_probing.info(info);
            NodeIterator itr = info["render_times"].children();
            while (itr.has_next())
            {
                Node &t = itr.next();
                render_times.push_back(t.to_double());
            }

            render_chunks["depths"].set_external(info["depths"]);
            render_chunks["color_buffers"].set_external(info["color_buffers"]);
            render_chunks["depth_buffers"].set_external(info["depth_buffers"]);
            render_chunks["render_file_names"].set_external(info["render_file_names"]);

            int probing_images = int(std::round(probing_factor * phi * theta));
            log_time(start, "probing " + std::to_string(probing_images) + " ", world_rank);

            std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
            total_probing_time = elapsed.count();
        }
    }
    else
    {
        render_times.push_back(100.f); // dummy value for in transit only test
    }

    log_global_time("end probing", world_rank);
#if ASCENT_MPI_ENABLED
    if (!is_inline) 
    {
        MPI_Properties mpi_props(world_size, world_rank, sim_count, world_size - sim_count, 
                                 mpi_comm_world, vis_comm);
        RenderConfig render_cfg(phi*theta, probing_factor, vis_budget);
        if (world_rank == 0)
        {
            std::cout << "Probing " << render_cfg.probing_count << "/" << render_cfg.max_count
                    << " renders with stride " << render_cfg.probing_stride << std::endl; 
        }

        hybrid_render(mpi_props, render_cfg, render_times, total_probing_time, m_data, render_chunks);
    }
    ascent_probing.close();

    MPI_Group_free(&world_group);
    MPI_Group_free(&sim_group);
    MPI_Group_free(&vis_group);

    if (sim_comm != MPI_COMM_NULL)
    {
        MPI_Barrier(sim_comm);
        MPI_Comm_free(&sim_comm);
    }
    else if (vis_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&vis_comm);
    }

    log_global_time("end ascent", world_rank);
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
