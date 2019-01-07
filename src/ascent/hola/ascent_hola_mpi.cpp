//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: ascent_hola_mpi.cpp
///
//-----------------------------------------------------------------------------

#include <hola/ascent_hola.hpp>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include <conduit_relay_mpi.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

#include <fstream>

using namespace conduit;
using namespace std;
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


//-----------------------------------------------------------------------------
int32
calc_offsets(const int32_array &lst,
             int32_array &offsets)
{
    int32 count = 0;
    for(int i=0;i<lst.number_of_elements();i++)
    {
        offsets[i] = count;
        count += lst[i];
    }

    return count;
}

//-----------------------------------------------------------------------------
void
gen_dest_domain_lst(const int32_array &lst,
                    int32 total_num_doms,
                    int32 dest_size,
                    int32_array &res)
{
    int32 doms_per_dest = total_num_doms / dest_size;
    int32 count = 0;
    for(int i=0; i < dest_size ;i++)
    {
        count += doms_per_dest;
        if(i == dest_size - 1)
        {
            doms_per_dest += total_num_doms - count;
        }
        res[i] = doms_per_dest;
    }
}

//-----------------------------------------------------------------------------
void
hola_mpi_comm_map(const conduit::Node &data,
                  MPI_Comm comm,
                  const conduit::int32_array &world_to_src,
                  const conduit::int32_array &world_to_dest,
                  conduit::Node &res)
{
    // calc src_size and dest_size

    int rank = relay::mpi::rank(comm);
    int total_size = relay::mpi::size(comm);
    int src_size  = 0;
    int dest_size = 0;

    for(int i=0; i < total_size; i++)
    {
        if(world_to_src[i] >=0)
        {
            src_size+=1;
        }

        if(world_to_dest[i] >=0)
        {
            dest_size+=1;
        }
    }

    bool is_source_rank = world_to_src[rank] >= 0;

    res.reset();

    // maps from src and dest index spack to world ranks

    res["src_to_world"]  = DataType::int32(src_size);
    res["dest_to_world"] = DataType::int32(dest_size);

    int32 *src_to_world  = res["src_to_world"].value();
    int32 *dest_to_world = res["dest_to_world"].value();

    int src_idx  = 0;
    int dest_idx = 0;

    for(int i=0; i < world_to_src.number_of_elements(); i++)
    {
        if(world_to_src[i] >=0)
        {
            src_to_world[src_idx] = i;
            src_idx++;
        }

        if(world_to_dest[i] >=0)
        {
            dest_to_world[dest_idx] = i;
            dest_idx++;
        }
    }

    Node n_num_loc_doms, n_num_total_doms;

    if(is_source_rank)
    {
        n_num_loc_doms.set_int32(data.number_of_children());
    }
    else
    {
        n_num_loc_doms.set_int32(0);
    }

    relay::mpi::all_gather_using_schema(n_num_loc_doms,
                                        n_num_total_doms,
                                        comm);

    Node n_num_total_doms_acc;

    n_num_total_doms_acc.set_external((int32*)n_num_total_doms.data_ptr(),total_size);
    int32_array num_doms = n_num_total_doms_acc.as_int32_array();

    res["src_counts"] = DataType::int32(src_size);
    int32_array src_counts =res["src_counts"].value();

    for(int i=0;i<src_size;i++)
    {
        src_counts[i] = num_doms[src_to_world[i]];
    }

    res["src_offsets"] = DataType::int32(src_size);

    int32_array src_offsets = res["src_offsets"].as_int32_array();
    int32 num_total_doms = calc_offsets(src_counts,
                                        src_offsets);

    res["dest_counts"] = DataType::int32(dest_size);

    int32_array dest_lst =  res["dest_counts"].as_int32_array();

    gen_dest_domain_lst(src_counts,
                        num_total_doms,
                        dest_size,
                        dest_lst);

    res["dest_offsets"] = DataType::int32(dest_size);
    int32_array dest_offsets = res["dest_offsets"].as_int32_array();
    int32 num_dest_domains = calc_offsets(dest_lst,
                                          dest_offsets);
}


//-----------------------------------------------------------------------------
void
hola_mpi_send(const conduit::Node &data,
              MPI_Comm comm,
              int src_idx,
              const conduit::Node &comm_map)
{
    const int32 *src_counts  = comm_map["src_counts"].value();
    const int32 *src_offsets = comm_map["src_offsets"].value();

    const int32 *dest_counts   = comm_map["dest_counts"].value();
    const int32 *dest_offsets  = comm_map["dest_offsets"].value();
    const int32 *dest_to_world = comm_map["dest_to_world"].value();

    // responsible for sending src_offsets[src_idx] + src_counts[src_idx]
    // to who ever needs them
    // assumes multi domain mesh bp
    // we expect: data.number_of_children() == src_counts[src_idx]
    NodeConstIterator itr = data.children();

    int dest_idx = 0;
    for(int i = src_offsets[src_idx];
        i < src_offsets[src_idx] + src_counts[src_idx];
        i++)
    {
        const Node &n_curr = itr.next();
        // find  i's dest
        while( i >= dest_offsets[dest_idx] + dest_counts[dest_idx])
        {
            dest_idx++;
        }

        int32 dest_rank = dest_to_world[(int32)dest_idx];

        // std::cout << "src_idx " << src_idx << " send " << i << " to "
        //           << dest_idx << " (rank: " << dest_rank <<  " )" << std::endl;

        relay::mpi::send_using_schema(n_curr,dest_rank,0,comm);
    }

}

//-----------------------------------------------------------------------------
void
hola_mpi_recv(MPI_Comm comm,
              int dest_idx,
              const conduit::Node &comm_map,
              conduit::Node &data)
{
    const int32 *src_counts  = comm_map["src_counts"].value();
    const int32 *src_offsets = comm_map["src_offsets"].value();
    const int32 *src_to_world = comm_map["src_to_world"].value();

    const int32 *dest_counts  = comm_map["dest_counts"].value();
    const int32 *dest_offsets = comm_map["dest_offsets"].value();

    // responsible for receiving dest_offsets[dest_idx] + dest_counts[dest_idx]
    // from who ever has them
    int src_idx = 0;
    for(int i = dest_offsets[dest_idx];
        i < dest_offsets[dest_idx] + dest_counts[dest_idx];
        i++)
    {
        // find  i's src
        while( i >= src_offsets[src_idx] + src_counts[src_idx])
        {
            src_idx++;
        }

        int32 src_rank = src_to_world[(int32)src_idx];
        // std::cout << "dest_idx " << dest_idx << " rcv "
        //           << i <<  " from " << src_idx << " ( rank: " << src_rank << ") " <<std::endl;
        Node &n_curr = data.append();
        // rcv n_curr
        relay::mpi::recv_using_schema(n_curr,src_rank,0,comm);
    }
}


//-----------------------------------------------------------------------------
void
hola_mpi(const conduit::Node &options,
         conduit::Node &data)
{
    MPI_Comm comm  = MPI_Comm_f2c(options["mpi_comm"].to_int());
    // get my rank
    int rank = relay::mpi::rank(comm);
    // get total size
    int total_size = relay::mpi::size(comm);

    int rank_split = options["rank_split"].to_int();

    //
    // TODO: We can enhance to also support the case
    // where client code passes in world to src and world to dest maps
    //

    // calc source size using split

    // source ranks are 0 to rank_split - 1
    int src_size = rank_split;

    // calc dest size using split
    // dest ranks are rank_split  to total_size -1
    int dest_size = total_size - rank_split;


    Node my_maps;
    my_maps["wts"] = DataType::int32(total_size);
    my_maps["wtd"] = DataType::int32(total_size);

    int32_array world_to_src  = my_maps["wts"].value();
    int32_array world_to_dest = my_maps["wtd"].value();

    for(int i=0;i<total_size;i++)
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

    // we are a src if world to src is not neg
    bool is_src_rank = world_to_src[rank] >= 0;

    // if sender, make sure we have multi domain data
    // if not, create a new node that zero copies the input data
    // into a multi domain layout
    Node *data_ptr = &data;
    Node md_data;

    if(is_src_rank && !blueprint::mesh::is_multi_domain(data))
    {
        md_data.append().set_external(data);
        data_ptr = &md_data;
    }

    Node comm_map;

    hola_mpi_comm_map(*data_ptr,
                      comm,
                      world_to_src,
                      world_to_dest,
                      comm_map);

    if(is_src_rank )
    {
        int src_idx = world_to_src[rank];
        hola_mpi_send(*data_ptr,comm,src_idx,comm_map);
    }
    else
    {
        int dest_idx = world_to_dest[rank];
        hola_mpi_recv(comm,dest_idx,comm_map,*data_ptr);
    }
}

//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------


