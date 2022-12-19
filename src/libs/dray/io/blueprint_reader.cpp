// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_low_order.hpp>

#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/utils/data_logger.hpp>

#ifdef DRAY_MFEM_ENABLED
#include <dray/mfem2dray.hpp>
#include <mfem/fem/conduitdatacollection.hpp>
#endif

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <conduit_relay_io_blueprint.hpp>

#include <fstream>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

using namespace conduit;

namespace dray
{
namespace detail
{

std::string append_cycle (const std::string &base, const int cycle)
{
  std::ostringstream oss;

  char fmt_buff[64];
  snprintf (fmt_buff, sizeof (fmt_buff), "%06d", cycle);
  oss.str ("");
  oss << base << "_" << std::string (fmt_buff);
  return oss.str ();
}

void make_domain_ids(conduit::Node &domains)
{
  int num_domains = domains.number_of_children();

  int domain_offset = 0;

#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  int comm_size = dray::mpi_size();
  int rank = dray::mpi_rank();

  int *domains_per_rank = new int[comm_size];

  MPI_Allgather(&num_domains, 1, MPI_INT, domains_per_rank, 1, MPI_INT, mpi_comm);

  for(int i = 0; i < rank; ++i)
  {
    domain_offset += domains_per_rank[i];
  }
  delete[] domains_per_rank;
#endif

  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = domains.child(i);
    dom["state/domain_id"] = domain_offset + i;
  }
}

class BlueprintTreePathGenerator
{
  public:
  BlueprintTreePathGenerator (const std::string &file_pattern,
                              const std::string &tree_pattern,
                              int num_files,
                              int num_trees,
                              const std::string &protocol,
                              const Node &mesh_index)
  : m_file_pattern (file_pattern), m_tree_pattern (tree_pattern),
    m_num_files (num_files), m_num_trees (num_trees), m_protocol (protocol),
    m_mesh_index (mesh_index)
  {
    (void) m_num_files;
    (void) m_num_trees;
  }

  //-------------------------------------------------------------------//
  ~BlueprintTreePathGenerator ()
  {
  }

  //-------------------------------------------------------------------//
  std::string Expand (const std::string pattern, int idx) const
  {
    //
    // Note: This currently only handles format strings:
    // "%05d" "%06d" "%07d"
    //

    std::size_t pattern_idx = pattern.find ("%05d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%05d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }

    pattern_idx = pattern.find ("%06d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%06d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }

    pattern_idx = pattern.find ("%07d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%07d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }
    return pattern;
  }


  //-------------------------------------------------------------------//
  std::string GenerateFilePath (int tree_id) const
  {
    // for now, we only support 1 tree per file.
    int file_id = tree_id;
    return Expand (m_file_pattern, file_id);
  }

  //-------------------------------------------------------------------//
  std::string GenerateTreePath (int tree_id) const
  {
    // the tree path should always end in a /
    std::string res = Expand (m_tree_pattern, tree_id);
    if ((res.size () > 0) && (res[res.size () - 1] != '/'))
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


bool is_high_order(const conduit::Node &domain)
{
  if(domain.has_path("fields"))
  {
    const conduit::Node &fields = domain["fields"];
    const int num_fields= fields.number_of_children();
    for(int t = 0; t < num_fields; ++t)
    {
      const conduit::Node &field = fields.child(t);
      if(field.has_path("basis")) return true;
    }
  }

  return false;
}

template <typename T>
DataSet bp_ho_2dray (const conduit::Node &n_dataset)
{
#ifndef DRAY_MFEM_ENABLED
    DRAY_ERROR("High-order Blueprint import requires MFEM, but DRay lacks"
               " MFEM support (MFEM_FOUND=FALSE)");
#else
  mfem::Mesh *mfem_mesh_ptr = mfem::ConduitDataCollection::BlueprintMeshToMesh (n_dataset);
  mfem::Geometry::Type geom_type = mfem_mesh_ptr->GetElementBaseGeometry(0);

  mfem_mesh_ptr->GetNodes ();

  DataSet dataset = import_mesh(*mfem_mesh_ptr);

  if(n_dataset.has_path("state/domain_id"))
  {
    dataset.domain_id(n_dataset["state/domain_id"].to_int32());
  }

  NodeConstIterator itr = n_dataset["fields"].children ();

  std::string nodes_gf_name = "";
  std::string topo_name = "main";

  if (n_dataset["topologies"].number_of_children () == 0)
  {
    // this should not happen if verify is called before
    DRAY_ERROR ("Blueprint dataset has no topologies");
  }
  else
  {
    std::vector<std::string> names = n_dataset["topologies"].child_names ();
    topo_name = names[0];
    dataset.mesh()->name(topo_name);
  }
#warning "should we import the boundry topology?"

  DRAY_INFO ("Found topology "<<topo_name);

  const Node &n_topo = n_dataset["topologies/" + topo_name];
  if (n_topo.has_child ("grid_function"))
  {
    nodes_gf_name = n_topo["grid_function"].as_string ();
  }

  DRAY_LOG_OPEN("import_fields");
  while (itr.has_next ())
  {
    const Node &n_field = itr.next ();
    std::string field_name = itr.name ();

    // skip mesh nodes gf since they are already processed
    // skip attribute fields, they aren't grid functions
    if (field_name != nodes_gf_name && field_name.find ("_attribute") == std::string::npos)
    {
      mfem::GridFunction *grid_ptr =
      mfem::ConduitDataCollection::BlueprintFieldToGridFunction (mfem_mesh_ptr, n_field);
      const mfem::FiniteElementSpace *fespace = grid_ptr->FESpace ();
      const int32 P = fespace->GetOrder (0);

      const int components = grid_ptr->VectorDim ();
      bool success = true;
      if (components == 1)
      {
        DRAY_INFO("Importing field "<<field_name);

        try
        {
          import_field(dataset, *grid_ptr, geom_type, field_name);
        }
        catch(const DRayError &e)
        {
          success = false;
          DRAY_WARN("field import '"<<field_name<<"' failed with error '"
                    <<e.what()<<"'");
        }
      }
      else if (components == 3 || components == 2)
      {
        try
        {
          import_vector(dataset, *grid_ptr, geom_type, field_name);
        }
        catch(const DRayError &e)
        {
          success = false;
          DRAY_WARN("vector field import '"<<field_name<<"' failed with error '"
                    <<e.what()<<"'");
        }
      }
      else
      {
        success = false;
        DRAY_INFO ("Import field '"<<field_name<<"': number of components = "
                   << components << " not supported");
      }
      delete grid_ptr;
      if(success)
      {
        DRAY_INFO ("Imported field name " << field_name);
      }
    }
  }
  DRAY_LOG_CLOSE();
  delete mfem_mesh_ptr;
  return dataset;
#endif
}

//-----------------------------------------------------------------------------

template <typename T>
DataSet bp2dray (const conduit::Node &n_domain)
{
  DataSet dataset;
  if(is_high_order(n_domain))
  {
    dataset = bp_ho_2dray<T>(n_domain);
  }
  else
  {
    dataset = BlueprintLowOrder::import(n_domain);
  }
  return dataset;
}

Collection load_bp(const std::string &root_file)
{
  DRAY_LOG_OPEN("load_bp");
  Node options, data;
  options["root_file"] = root_file;
  conduit::relay::io::blueprint::load_mesh(root_file, data);

  const int num_domains = data.number_of_children();
  Collection collection;
  DRAY_LOG_OPEN("convert_bp");
  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &domain = data.child(i);
    int domain_id = domain["state/domain_id"].to_int32();
    DRAY_INFO("Importing domain "<<domain_id);
    DataSet dset = bp2dray<Float> (domain);
    collection.add_domain(dset);
  }
  DRAY_LOG_CLOSE();
  DRAY_LOG_CLOSE();
  return collection;
}

} // namespace detail

void
BlueprintReader::load_blueprint(const std::string &root_file,
                                conduit::Node &dataset)
{
  conduit::Node options;
  options["root_file"] = root_file;
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  conduit::relay::mpi::io::blueprint::load_mesh(root_file, dataset, mpi_comm);
#else
  conduit::relay::io::blueprint::load_mesh(root_file, dataset);
#endif
}

void
BlueprintReader::save_blueprint(const std::string &root_file,
                                conduit::Node &dataset)
{
  // this is a mulit-dom data set

  // we might have removed some of the domain ids
  // or they might not exist
  detail::make_domain_ids(dataset);
  const int num_domains = dataset.number_of_children();
  int global_domains = num_domains;
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Allreduce((void *)(&num_domains),
                (void *)(&global_domains),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
#endif

  if(global_domains == 0)
  {
    DRAY_ERROR("There is no data");
  }
  // we may not have any domains so init to max
  int cycle = std::numeric_limits<int>::max();

  if(num_domains > 0)
  {
    conduit::Node dom = dataset.child(0);
    if(!dom.has_path("state/cycle"))
    {
      static std::map<std::string,int> counters;
      // Defaulting to counter
      cycle = counters[root_file];
      counters[root_file]++;
    }
    else
    {
      cycle = dom["state/cycle"].to_int();
    }
  }

#ifdef DRAY_MPI_ENABLED
  conduit::Node n_cycle, n_min;
  n_cycle = (int)cycle;

  relay::mpi::min_all_reduce(n_cycle,
                      n_min,
                      mpi_comm);

  cycle = n_min.as_int();
#endif
 // setup the directory
 char fmt_buff[64] = {0};
 snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

 std::string output_base_path = root_file;

 std::ostringstream oss;
 oss << output_base_path << ".cycle_" << fmt_buff;
 std::string output_dir  =  oss.str();

 bool dir_ok = false;

 // let rank zero handle dir creation
 if(dray::mpi_rank() == 0)
 {
   // check of the dir exists
   dir_ok = conduit::utils::is_directory(output_dir);
   if(!dir_ok)
   {
     // if not try to let rank zero create it
     dir_ok = conduit::utils::create_directory(output_dir);
   }
 }

 const std::string file_protocol = "hdf5";

 // write out each domain
 for(int i = 0; i < num_domains; ++i)
 {
   const Node &dom = dataset.child(i);
   uint64 domain = dom["state/domain_id"].to_uint64();

   snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
   oss.str("");
   oss << "domain_" << fmt_buff << "." << file_protocol;
   std::string output_file  = conduit::utils::join_file_path(output_dir,oss.str());
   relay::io::save(dom, output_file);
 }

  int root_file_writer = 0;
  if(num_domains == 0)
  {
    root_file_writer = -1;
  }
#ifdef DRAY_MPI_ENABLED
  // Rank 0 could have an empty domain, so we have to check
  // to find someone with a data set to write out the root file.
  conduit::Node out;
  out = num_domains;
  conduit::Node rcv;

  conduit::relay::mpi::all_gather_using_schema(out, rcv, mpi_comm);
  root_file_writer = -1;
  int* res_ptr = (int*)rcv.data_ptr();
  const int mpi_ranks = dray::mpi_size();
  for(int i = 0; i < mpi_ranks; ++i)
  {
    if(res_ptr[i] != 0)
    {
      root_file_writer = i;
      break;
    }
  }

  MPI_Barrier(mpi_comm);
#endif

 // let rank zero write out the root file
  if(dray::mpi_rank() == root_file_writer)
  {
    snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

    oss.str("");
    oss << root_file
        << ".cycle_"
        << fmt_buff
        << ".root";

    std::string root_file = oss.str();

    std::string output_dir_base, output_dir_path;

    // TODO: Fix for windows
    conduit::utils::rsplit_string(output_dir,
                                  "/",
                                  output_dir_base,
                                  output_dir_path);

    std::string output_tree_pattern;
    std::string output_file_pattern;

    output_tree_pattern = "/";
    output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                        "domain_%06d." + file_protocol);


    conduit::Node root;
    conduit::Node &bp_idx = root["blueprint_index"];

    blueprint::mesh::generate_index(dataset.child(0),
                                    "",
                                    global_domains,
                                    bp_idx["mesh"]);

    // work around conduit and manually add state fields
    if(dataset.child(0).has_path("state/cycle"))
    {
      bp_idx["mesh/state/cycle"] = dataset.child(0)["state/cycle"].to_int32();
    }

    if(dataset.child(0).has_path("state/time"))
    {
      bp_idx["mesh/state/time"] = dataset.child(0)["state/time"].to_double();
    }

    root["protocol/name"]    = file_protocol;
    root["protocol/version"] = "0.5.1";

    root["number_of_files"]  = global_domains;
    root["number_of_trees"]  = global_domains;
    // TODO: make sure this is relative
    root["file_pattern"]     = output_file_pattern;
    root["tree_pattern"]     = output_tree_pattern;

    conduit::relay::io::save(root,root_file,file_protocol);
  }
}

Collection BlueprintReader::load (const std::string &root_file)
{
  return detail::load_bp (root_file);
}

Collection BlueprintReader::load (const std::string &root_file, const int cycle)
{
  std::string full_root = detail::append_cycle (root_file, cycle) + ".root";
  return detail::load_bp (full_root);
}

DataSet
BlueprintReader::blueprint_to_dray (const conduit::Node &n_dataset)
{
  return detail::bp2dray<Float> (n_dataset);
}

} // namespace dray
