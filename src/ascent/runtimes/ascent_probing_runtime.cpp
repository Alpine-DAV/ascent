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
    Node n_src, n_reduce;

    if (verify_ok)
        n_src = (int)0;
    else
        n_src = (int)1;

    conduit::relay::mpi::sum_all_reduce(n_src,
                                        n_reduce,
                                        mpi_comm);

    int num_failures = n_reduce.value();
    if (num_failures != 0)
    {
        ASCENT_ERROR("Mesh Blueprint Verify failed on "
                     << num_failures
                     << " MPI Tasks");

        // you could use mpi to find out where things went wrong ...
    }

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
bool decide_intransit(const double avg, const float vis_budget)
{
    // TODO: calculate based on budget
    double max_time = 105.f;

    if (avg > max_time)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//-----------------------------------------------------------------------------
void splitAndRender(const MPI_Comm mpi_comm_world,
                    const std::vector<double> &render_times,
                    conduit::Node &data,
                    const int sim_node_count,
                    const double vis_budget = 0.1)
{
    assert(vis_budget > 0.0 && vis_budget < 1.0);

    int rank = -1;
    int total_ranks = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm_size(mpi_comm_world, &total_ranks);
    MPI_Comm_rank(mpi_comm_world, &rank);
#endif
    assert(sim_node_count > 0 && sim_node_count <= total_ranks);

    int is_intransit = 0;
    int is_rendering = 1;
    int is_sending = 0;
    bool is_vis_node = false;

    if (rank >= sim_node_count) // nodes with the highest rank are our vis nodes
    {
        // vis nodes are always in the in-transit comm
        is_intransit = 1;
        // vis nodes only receive and render data
        is_vis_node = true;
        std::cout << "~~~ "
                  << "rank " << rank << " is a vis node." << std::endl;
    }
    else if (total_ranks > 1)
    {
        assert(render_times.size() > 0);
        double avg = std::accumulate(render_times.begin(), render_times.end(), 0.0) / render_times.size();
        std::cout << "~~~ " << avg << " ms mean frame time rank " << rank << std::endl;

        // decide if this node wants to send data away
        if (decide_intransit(avg, vis_budget))
            is_intransit = 1;
    }

    // all sending in transit nodes
    if (is_intransit == 1 && !is_vis_node)
    {
        is_rendering = 0;
        is_sending = 1;
    }

#ifdef ASCENT_MPI_ENABLED
    // split the current comm into in-line and in-transit nodes
    MPI_Comm intransit_comm;
    MPI_Comm_split(mpi_comm_world, is_intransit, 0, &intransit_comm);
    int intransit_size = 0;
    MPI_Comm_size(intransit_comm, &intransit_size);
    // Hola setup
    MPI_Comm hola_comm;
    MPI_Comm_split(intransit_comm, is_sending, 0, &hola_comm);
    int rank_split = intransit_size - (total_ranks - sim_node_count);
    // split world comm into rendering and sending nodes
    MPI_Comm render_comm;
    MPI_Comm_split(mpi_comm_world, is_rendering, 0, &render_comm);

    if (is_rendering) // all rendering nodes
    {
        if (is_vis_node && rank_split > 0) // render on vis node
        {
            std::cout << "~~~~rank " << rank << ": receives extract(s)." << std::endl;

            // use hola to receive the extract data
            Node hola_opts;
            hola_opts["mpi_comm"] = MPI_Comm_c2f(intransit_comm);
            hola_opts["rank_split"] = rank_split;
            ascent::hola("hola_mpi", hola_opts, data);
        }
        else // render local (inline)
        {
            std::cout << "~~~~rank " << rank << ": renders inline." << std::endl;
        }
        // Render the data using Ascent
        MPI_Barrier(render_comm);

        Node ascent_opts, blank_actions;
        ascent_opts["actions_file"] = "cinema_actions.yaml";
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(render_comm);

        Ascent ascent_render;
        ascent_render.open(ascent_opts);
        ascent_render.publish(data);
        // FIXME: brakes here if inline rendering
        ascent_render.execute(blank_actions);

        // WARNING ascent.open(render_comm); sets the static variable that holds
        // the VTK-h mpi comm. That means you will have to set this back to what
        // it was before. This is probably one of the reasons you are deadlocked.
        ascent_render.close();

        // reset vtkh communicator
        // MPI_Barrier(mpi_comm_world);
        // vtkh::SetMPICommHandle(mpi_comm_world);

        std::cout << "----rank " << rank << ": ascent_render - finished."
                  << std::endl;
    }
    else // all nodes not rendering: send extract to vis nodes using Hola
    {
        std::cout << "~~~~rank " << rank << ": sends extract." << std::endl;
        // add the extract
        conduit::Node actions;
        conduit::Node &add_extract = actions.append();
        add_extract["action"] = "add_extracts";
        add_extract["extracts/e1/type"] = "hola_mpi";
        add_extract["extracts/e1/params/mpi_comm"] = MPI_Comm_c2f(intransit_comm);
        add_extract["extracts/e1/params/rank_split"] = rank_split;

        Node ascent_opts;
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(hola_comm);

        // Send an extract of the data with Ascent
        Ascent ascent_send;
        ascent_send.open(ascent_opts);
        ascent_send.publish(data);
        ascent_send.execute(actions); // extract
        ascent_send.close();
        std::cout << "----rank " << rank << ": ascent_send." << std::endl;
    }
    MPI_Barrier(mpi_comm_world);

    // FIXME: freeing the comms brakes execution ??
    MPI_Comm_free(&render_comm);
    MPI_Comm_free(&hola_comm);
    MPI_Comm_free(&intransit_comm);

    MPI_Barrier(mpi_comm_world);
#endif // ASCENT_MPI_ENABLED
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
                    probing_factor = action["probing/factor"].to_double();
                else
                    ASCENT_ERROR("action 'probing' missing child 'factor'");

                if (action["probing"].has_path("vis_budget"))
                    vis_budget = action["probing/vis_budget"].to_double();
                else
                    ASCENT_ERROR("action 'probing' missing child 'vis_budget'");

                if (action["probing"].has_path("node_split"))
                    node_split = action["probing/node_split"].to_double();
                else
                    ASCENT_ERROR("action 'probing' missing child 'node_split'");
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

    MPI_Comm sim_comm;
    MPI_Comm_split(comm_world, color, 0, &sim_comm);
    ascent_opt["mpi_comm"] = sim_comm;
#endif // ASCENT_MPI_ENABLED

    std::vector<double> render_times;
    // run probing only if this is a sim node (vis nodes don't have data yet)
    // TODO: we could check for data validity instead
    if (world_rank < rank_split)
    {
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime

        // all sim nodes run probing in a new ascent instance
        Ascent ascent_probing;
        ascent_probing.open(ascent_opt);
        ascent_probing.publish(m_data);        // pass on data pointer
        ascent_probing.execute(probe_actions); // pass on actions

        // TODO we need the rendering times from ascent: use .info() interface ?
        conduit::Node info;
        ascent_probing.info(info);

        NodeIterator itr = info["render_times"].children();
        while (itr.has_next())
        {
            Node &t = itr.next();
            render_times.push_back(t.to_double());
            // std::cout << t.to_double() << std::endl;
        }
        ascent_probing.close();
    }
#if ASCENT_MPI_ENABLED
    // MPI_Barrier(comm_world);
    // split comm into sim and vis nodes and render on the respective nodes
    splitAndRender(comm_world, render_times, m_data, rank_split, vis_budget);
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
