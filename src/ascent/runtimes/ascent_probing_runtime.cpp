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

//-----------------------------------------------------------------------------
void create_comm_maps(Node &my_maps, const int total_size, const int src_size)
{
    my_maps["wts"] = DataType::int32(total_size);
    my_maps["wtd"] = DataType::int32(total_size);

    int32_array world_to_src  = my_maps["wts"].value();
    int32_array world_to_dest = my_maps["wtd"].value();

    for(int i=0; i < total_size; i++)
    {
        if(i < src_size)
        {
            world_to_dest[i] = -1;
            world_to_src[i]  = i;
        }
        else
        {
            world_to_dest[i] = i - src_size;
            world_to_src[i] = -1;
        }
    }
}

//-----------------------------------------------------------------------------
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const int world_size,
                    const int world_rank,
                    const int sim_node_count,
                    const std::vector<double> &render_times,
                    conduit::Node &data,
                    const double vis_budget = 0.1)
{
    // assert(vis_budget > 0.0 && vis_budget < 1.0);    // currently absolute ms value
    assert(sim_node_count > 0 && sim_node_count <= world_size);

    int is_intransit = 0;
    int is_rendering = 1;
    int is_sending = 0;
    bool is_vis_node = false;

    float my_avg_probing_time = 0.f;

    // nodes with the highest rank are our vis nodes
    if (world_rank >= sim_node_count) 
    {
        // vis nodes are always in the in-transit comm
        is_intransit = 1;
        // vis nodes only receive and render data
        is_vis_node = true;
    }
    else if (world_size > 1)
    {
        assert(render_times.size() > 0);
        my_avg_probing_time = float(std::accumulate(render_times.begin(), render_times.end(), 0.0) 
                     / render_times.size());
        std::cout << "~~~ " << my_avg_probing_time << " ms mean frame time rank " 
                  << world_rank << std::endl;
    }

#ifdef ASCENT_MPI_ENABLED
    std::vector<float> probing_times(world_size);
    MPI_Allgather(&my_avg_probing_time, 1, MPI_FLOAT, 
                  probing_times.data(), 1, MPI_FLOAT, mpi_comm_world);
    // TODO: add global job assignment

    // decide if this node wants to send data away
    if (!is_vis_node && decide_intransit(probing_times, world_rank, vis_budget))
    {
        // all sending in-transit nodes
        is_intransit = 1;
        is_rendering = 0;
        is_sending = 1;
    }

    // TODO: change comm_split to comm_create_group
    // split the current comm into in-line and in-transit nodes
    MPI_Comm intransit_comm;
    MPI_Comm_split(mpi_comm_world, is_intransit, 0, &intransit_comm);
    int intransit_size = 0;
    // NOTE: implicit barrier (collective operation)? -> global sync anyway?
    MPI_Comm_size(intransit_comm, &intransit_size);

    // Node comm_maps;
    // if (is_intransit)
    //     create_comm_maps(&comm_maps, probing_times);

    int vis_node_count = world_size - sim_node_count;
    int sending_node_count = intransit_size - vis_node_count;

    if (is_vis_node)
    {
        int my_vis_rank = world_rank - sim_node_count;
        // if less sim nodes than vis nodes sends data
        // vis nodes skip rendering accordingly
        if (sending_node_count <= my_vis_rank)
        {
            is_rendering = 0;
            std::cout << "~~~~rank " << world_rank << ": idles." << std::endl;
            // FIXME: deadlock if #vis_nodes > #sending_nodes > 0
        }
    }

    // Hola setup
    MPI_Comm hola_comm;
    MPI_Comm_split(intransit_comm, is_sending, 0, &hola_comm);
    // split world comm into rendering and sending nodes
    MPI_Comm render_comm;
    MPI_Comm_split(mpi_comm_world, is_rendering, 0, &render_comm);

    if (is_rendering) // all rendering nodes
    {
        if (is_vis_node) // render on vis node
        {
            std::cout << "~~~~rank " << world_rank << ": receives extract(s)." << std::endl;
            // use hola to receive the extract data
            Node hola_opts;
            hola_opts["mpi_comm"] = MPI_Comm_c2f(intransit_comm);
            hola_opts["rank_split"] = sending_node_count;
            // TODO: add custom comm map
            // hola_opts["comm_maps"] = comm_maps;

            ascent::hola("hola_mpi", hola_opts, data);
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
            ascent_render.publish(data);
            ascent_render.execute(blank_actions);
            ascent_render.close();
        }
        else
        {
            std::cout << "~~~~rank " << world_rank << ": could not verify sent data." 
                      << std::endl;
        }
    }
    else if (!is_vis_node) // all sending nodes: send extract to vis nodes using Hola
    {
        std::cout << "~~~~rank " << world_rank << ": sends extract." << std::endl;
        // add the extract
        conduit::Node actions;
        conduit::Node &add_extract = actions.append();
        add_extract["action"] = "add_extracts";
        add_extract["extracts/e1/type"] = "hola_mpi";
        add_extract["extracts/e1/params/mpi_comm"] = MPI_Comm_c2f(intransit_comm);
        add_extract["extracts/e1/params/rank_split"] = sending_node_count;

        Node ascent_opts;
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(hola_comm);

        // Send an extract of the data with Ascent
        Ascent ascent_send;
        ascent_send.open(ascent_opts);
        ascent_send.publish(data);
        ascent_send.execute(actions); // extract
        ascent_send.close();
        // std::cout << "----rank " << world_rank << ": ascent_send." << std::endl;
    }

    // clean up the split comms
    MPI_Comm_free(&render_comm);
    MPI_Comm_free(&hola_comm);
    MPI_Comm_free(&intransit_comm);
#endif // ASCENT_MPI_ENABLED
}

std::string get_timing_file_name(const int value, const int precision)
{
    std::ostringstream oss;
    oss << "timings/ascent_";
    oss << std::setw(precision) << std::setfill('0') << value;
    oss << ".txt";
    return oss.str();
}

void log_time(std::chrono::time_point<std::chrono::system_clock> start, int rank)
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count()
    //           << "s rank " << rank << std::endl;
    std::ofstream out(get_timing_file_name(rank, 5), std::ios_base::app);
    out << elapsed.count() << std::endl;
    out.close();
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Execute(const conduit::Node &actions)
{
    int world_rank = 0;
    int world_size = 1;
#if ASCENT_MPI_ENABLED
    // split comm into sim and vis nodes
    MPI_Comm comm_world = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &world_size);
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

    auto start = std::chrono::system_clock::now();

    int rank_split = 0;
#if ASCENT_MPI_ENABLED
    rank_split = int(std::round(world_size * node_split));
    int color = 0;
    if (world_rank >= rank_split)
        color = 1;

    MPI_Comm sim_comm;
    MPI_Comm_split(comm_world, color, 0, &sim_comm);
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
    splitAndRender(comm_world, world_size, world_rank, rank_split, 
                   render_times, m_data, vis_budget);

    log_time(start, world_rank);
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
