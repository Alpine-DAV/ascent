//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
#include <conduit_relay_io_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

#include <fstream>

#if defined(ASCENT_MPI_ENABLED)
    #include "ascent_hola_mpi.hpp"
    #include <conduit_relay_mpi.hpp>
    #include <conduit_relay_mpi_io_blueprint.hpp>
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
void hola(const std::string &source,
          const Node &options,
          Node &data)
{
    data.reset();
    if(source == "relay/blueprint/mesh")
    {
	std::string root_file = options["root_file"].as_string();
#if defined(ASCENT_MPI_ENABLED)
	MPI_Comm comm  = MPI_Comm_f2c(options["mpi_comm"].to_int());
	conduit::relay::mpi::io::blueprint::load_mesh(root_file,data,comm);
#else
	conduit::relay::io::blueprint::load_mesh(root_file,data);
#endif
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


