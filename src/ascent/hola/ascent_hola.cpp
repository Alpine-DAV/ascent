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
/// file: ascent_hola.cpp
///
//-----------------------------------------------------------------------------

#include <ascent_hola.hpp>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include <conduit_relay_io.hpp>
#include <conduit_relay_io_handle.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

#include <fstream>

#if defined(ASCENT_MPI_ENABLED)
    #include "ascent_hola_mpi.hpp"
    #include <conduit_relay_mpi.hpp>
#endif

using namespace conduit;
using namespace std;
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

// todo: this belongs in conduit relay io

class BlueprintTreePathGenerator
{
public:
    //-------------------------------------------------------------------//
    BlueprintTreePathGenerator(const std::string &file_pattern,
                               const std::string &tree_pattern,
                               int num_files,
                               int num_trees,
                               const std::string &protocol,
                               const Node &mesh_index)
    : m_file_pattern(file_pattern),
      m_tree_pattern(tree_pattern),
      m_num_files(num_files),
      m_num_trees(num_trees),
      m_protocol(protocol),
      m_mesh_index(mesh_index)
    {

    }

    //-------------------------------------------------------------------//
    ~BlueprintTreePathGenerator()
    {

    }

    //-------------------------------------------------------------------//
    std::string Expand(const std::string pattern,
                       int idx) const
    {
        //
        // Note: This currently only handles format strings:
        // "%05d" "%06d" "%07d"
        //

        std::size_t pattern_idx = pattern.find("%05d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%05d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }

        pattern_idx = pattern.find("%06d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%06d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }

        pattern_idx = pattern.find("%07d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%07d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }
        return pattern;
    }

    void gen_domain_to_file_map(int num_domains,
                                int num_files,
                                Node &out) const
    {
        int num_domains_per_file = num_domains / num_files;
        int left_overs = num_domains % num_files;

        out["global_domains_per_file"].set(DataType::int32(num_files));
        out["global_domain_offsets"].set(DataType::int32(num_files));
        out["global_domain_to_file"].set(DataType::int32(num_domains));

        int32_array v_domains_per_file = out["global_domains_per_file"].value();
        int32_array v_domains_offsets  = out["global_domain_offsets"].value();
        int32_array v_domain_to_file   = out["global_domain_to_file"].value();

        // setup domains per file
        for(int f=0; f < num_files; f++)
        {
            v_domains_per_file[f] = num_domains_per_file;
            if( f < left_overs)
                v_domains_per_file[f]+=1;
        }

        // prefix sum to calc offsets
        for(int f=0; f < num_files; f++)
        {
            v_domains_offsets[f] = v_domains_per_file[f];
            if(f > 0)
                v_domains_offsets[f] += v_domains_offsets[f-1];
        }

        // do assignment, create simple map
        int f_idx = 0;
        for(int d=0; d < num_domains; d++)
        {
            if(d >= v_domains_offsets[f_idx])
                f_idx++;
            v_domain_to_file[d] = f_idx;
        }
    }
    //-------------------------------------------------------------------//
    std::string GenerateFilePath(int tree_id) const
    {
        int file_id = -1;

        if(m_num_trees == m_num_files)
        {
            file_id = tree_id;
        }
        else if(m_num_files == 1)
        {
            file_id = 0;
        }
        else
        {
            Node d2f_map;
            gen_domain_to_file_map(m_num_trees,
                                    m_num_files,
                                    d2f_map);
            int num_domains_per_file = m_num_trees / m_num_files;
            int left_overs = m_num_trees % m_num_files;
            int32_array v_domain_to_file = d2f_map["global_domain_to_file"].value();
            file_id = v_domain_to_file[tree_id];
        }

        return Expand(m_file_pattern,file_id);
    }

    //-------------------------------------------------------------------//
    std::string GenerateTreePath(int tree_id) const
    {
        // the tree path should always end in a /
        std::string res = Expand(m_tree_pattern,tree_id);
        if( (res.size() > 0) && (res[res.size()-1] != '/') )
        {
            res += "/";
        }
        return res;
    }

private:
    std::string m_file_pattern;
    std::string m_tree_pattern;
    int m_num_files;
    int m_num_trees;
    std::string m_protocol;
    Node m_mesh_index;

};

//-----------------------------------------------------------------------------
void relay_blueprint_mesh_read(const Node &options,
                               Node &data)
{
    std::string root_fname = options["root_file"].as_string();

    // read the root file, it can be either json or hdf5

    // assume hdf5, but check for json file
    std::string root_protocol = "hdf5";

    // we will read only 5 bytes + keep the buffer null termed.
    char buff[6] = {0,0,0,0,0,0};

    // heuristic, if json, we expect to see "{" in the first 5 chars of the file.
    ifstream ifs;
    ifs.open(root_fname.c_str());
    if(!ifs.is_open())
    {
       ASCENT_ERROR("failed to open relay root file: " << root_fname);
    }
    ifs.read((char *)buff,5);
    ifs.close();

    std::string test_str(buff);

    if(test_str.find("{") != std::string::npos)
    {
       root_protocol = "json";
    }

    Node root_node;
    relay::io::load(root_fname, root_protocol, root_node);


    if(!root_node.has_child("file_pattern"))
    {
        ASCENT_ERROR("Root file missing 'file_pattern'");
    }

    if(!root_node.has_child("blueprint_index"))
    {
        ASCENT_ERROR("Root file missing 'blueprint_index'");
    }

    NodeConstIterator itr = root_node["blueprint_index"].children();
    Node verify_info;
    // TODO, for now lets verify the first mesh index

    const Node &mesh_index = itr.next();

    if( !blueprint::mesh::index::verify(mesh_index,
                                        verify_info[itr.name()]))
    {
        ASCENT_ERROR("Mesh Blueprint index verify failed" << std::endl
                     << verify_info.to_json());
    }

    std::string data_protocol = "hdf5";

    if(root_node.has_child("protocol"))
    {
        data_protocol = root_node["protocol/name"].as_string();
    }

    // read the first mesh (all domains ...)

    int num_domains = root_node["number_of_trees"].to_int();

    BlueprintTreePathGenerator gen(root_node["file_pattern"].as_string(),
                                   root_node["tree_pattern"].as_string(),
                                   root_node["number_of_files"].to_int(),
                                   num_domains,
                                   data_protocol,
                                   mesh_index);

    std::ostringstream oss;

    int domain_start = 0;
    int domain_end = num_domains;

#if defined(ASCENT_MPI_ENABLED)
    MPI_Comm comm  = MPI_Comm_f2c(options["mpi_comm"].to_int());
    int rank = relay::mpi::rank(comm);
    int total_size = relay::mpi::size(comm);

    int read_size = num_domains / total_size;
    int rem = num_domains % total_size;
    if(rank < rem)
    {
      read_size++;
    }

    conduit::Node n_read_size;
    conduit::Node n_doms_per_rank;

    n_read_size.set_int32(read_size);

    relay::mpi::all_gather_using_schema(n_read_size,
                                        n_doms_per_rank,
                                        comm);
    int *counts = (int*)n_doms_per_rank.data_ptr();

    int rank_offset = 0;
    for(int i = 0; i < rank; ++i)
    {
      rank_offset += counts[i];
    }

    domain_start = rank_offset;
    domain_end = rank_offset + read_size;
#endif

    relay::io::IOHandle hnd;
    for(int i = domain_start ; i < domain_end; i++)
    {
      char domain_fmt_buff[64];
      snprintf(domain_fmt_buff, sizeof(domain_fmt_buff), "%06d",i);
      oss.str("");
      oss << "domain_" << std::string(domain_fmt_buff);

      std::string current, next;
      utils::rsplit_file_path (root_fname, current, next);
      std::string domain_file = utils::join_path(next, gen.GenerateFilePath(i));

      hnd.open(domain_file, data_protocol);

      // also need the tree path
      std::string tree_path = gen.GenerateTreePath(i);

      //std::string mesh_path = conduit_fmt::format("domain_{:06d}",i);
      std::string mesh_path = oss.str();

      Node &mesh_out = data[mesh_path];

      // read components of the mesh according to the mesh index
      // for each child in the index
      NodeConstIterator outer_itr = mesh_index.children();
      while(outer_itr.has_next())
      {
        const Node &outer = outer_itr.next();
        std::string outer_name = outer_itr.name();

        // special logic for state, since it was not included in the index
        if(outer_name == "state" )
        {
          // we do need to read the state!
          if(outer.has_child("path"))
          {
            hnd.read(utils::join_path(tree_path,outer["path"].as_string()),
                     mesh_out[outer_name]);
          }
          else
          {
             if(outer.has_child("cycle"))
             {
                mesh_out[outer_name]["cycle"] = outer["cycle"];
             }

             if(outer.has_child("time"))
             {
               mesh_out[outer_name]["time"] = outer["time"];
             }
           }
        }

        NodeConstIterator itr = outer.children();
        while(itr.has_next())
        {
          const Node &entry = itr.next();
          // check if it has a path
          if(entry.has_child("path"))
          {
            std::string entry_name = itr.name();
            std::string entry_path = entry["path"].as_string();
            std::string fetch_path = utils::join_path(tree_path,
                                                      entry_path);
            // some parts may not exist in all domains
            // only read if they are there
            if(hnd.has_path(fetch_path))
            {
              hnd.read(fetch_path,
                       mesh_out[outer_name][entry_name]);
            }
          }
        }
      }
    }
}

//-----------------------------------------------------------------------------
void hola(const std::string &source,
          const Node &options,
          Node &data)
{
    data.reset();
    if(source == "relay/blueprint/mesh")
    {
        relay_blueprint_mesh_read(options,data);
    }
    else if(source == "hola_mpi")
    {
#if defined(ASCENT_MPI_ENABLED)
        hola_mpi(options,data);
#else
        ASCENT_ERROR("mpi disabled: 'hola_mpi' can only be used in ascent_mpi" );
#endif
    }
    else
    {
        ASCENT_ERROR("Unknown hola source: " << source);
    }

}

//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------


