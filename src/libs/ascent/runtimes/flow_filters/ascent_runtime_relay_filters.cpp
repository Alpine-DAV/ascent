//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_relay_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_relay_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mesh.hpp>
#include <conduit_relay_io_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_metadata.hpp>
#include <ascent_mpi_utils.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_runtime_param_check.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#include <conduit_blueprint_mpi_mesh.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

// std includes
#include <limits>
#include <set>

using namespace std;
using namespace conduit;
using namespace conduit::relay;
using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::detail --
//-----------------------------------------------------------------------------
namespace detail
{


//-----------------------------------------------------------------------------
// mfem needs special fields so look for them
//-----------------------------------------------------------------------------
void
check_for_attributes(const conduit::Node &input,
                     std::vector<std::string> &list)
{
  const int num_doms = input.number_of_children();
  std::set<std::string> specials;
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      std::vector<std::string> fnames = fields.child_names();
      for(int i = 0; i < fnames.size(); ++i)
      {
        if(fnames[i].find("_attribute") != std::string::npos)
        {
          specials.insert(fnames[i]);
        }
      }
    }
  }

  for(auto it = specials.begin(); it != specials.end(); ++it)
  {
    list.push_back(*it);
  }
}

//-----------------------------------------------------------------------------
void
filter_fields(const conduit::Node &input,
              conduit::Node &output,
              std::vector<std::string> fields,
              flow::Graph &graph)
{
  // assume this is multi-domain
  //
  check_for_attributes(input, fields);

  const int num_doms = input.number_of_children();
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    conduit::Node &out_dom = output.append();
    std::set<std::string> topos;
    std::set<std::string> matsets;

    for(int f = 0; f < fields.size(); ++f)
    {
      const std::string fname = fields[f];
      if(dom.has_path("fields/" + fname))
      {
        const std::string fpath = "fields/" + fname;
        out_dom[fpath].set_external(dom[fpath]);
        // check for topologies
        const std::string topo = dom[fpath + "/topology"].as_string();
        const std::string tpath = "topologies/" + topo;
        topos.insert(topo);

        // check for matset
        if(dom.has_path(fpath + "/matset"))
        {
          const std::string mopo = dom[fpath + "/matset"].as_string();
          matsets.insert(mopo);
        }

        if(!out_dom.has_path(tpath))
        {
          out_dom[tpath].set_external(dom[tpath]);
          if(dom.has_path(tpath + "/grid_function"))
          {
            const std::string gf_name = dom[tpath + "/grid_function"].as_string();
            const std::string gf_path = "fields/" + gf_name;
            out_dom[gf_path].set_external(dom[gf_path]);
          }
          if(dom.has_path(tpath + "/boundary_topology"))
          {
            const std::string bname = dom[tpath + "/boundary_topology"].as_string();
            const std::string bpath = "topologies/" + bname;
            out_dom[bpath].set_external(dom[bpath]);
          }
        }
        // check for coord sets
        const std::string coords = dom[tpath + "/coordset"].as_string();
        const std::string cpath = "coordsets/" + coords;
        if(!out_dom.has_path(cpath))
        {
          out_dom[cpath].set_external(dom[cpath]);
        }
      }

    }

    // add nestsets associated with referenced topologies
    if(dom.has_path("nestsets"))
    {
      const int num_nestsets = dom["nestsets"].number_of_children();
      const std::vector<std::string> nest_names = dom["nestsets"].child_names();
      for(int i = 0; i < num_nestsets; ++i)
      {
        const conduit::Node &nestset = dom["nestsets"].child(i);
        const std::string nest_topo = nestset["topology"].as_string();
        if(topos.find(nest_topo) != topos.end())
        {
          out_dom["nestsets/"+nest_names[i]].set_external(nestset);
        }
      }
    }

    // add nestsets associated with referenced topologies
    if(dom.has_path("matsets"))
    {
      const int num_matts = dom["matsets"].number_of_children();
      const std::vector<std::string> matt_names = dom["matsets"].child_names();
      for(int i = 0; i < num_matts; ++i)
      {
        const conduit::Node &matt = dom["matsets"].child(i);
        if(matsets.find(matt_names[i]) != matsets.end())
        {
          out_dom["matsets/"+matt_names[i]].set_external(matt);
        }
      }
    }

    // auto save out ghost fields from subset of topologies
    Node meta = Metadata::n_metadata;
    if(meta.has_path("ghost_field"))
    {
      const conduit::Node ghost_list = meta["ghost_field"];
      const int num_ghosts = ghost_list.number_of_children();

      for(int i = 0; i < num_ghosts; ++i)
      {
        std::string ghost = ghost_list.child(i).as_string();

        if(dom.has_path("fields/"+ghost) &&
           !out_dom.has_path("fields/"+ghost))
        {
          const conduit::Node &ghost_field = dom["fields/"+ghost];
          const std::string ghost_topo = ghost_field["topology"].as_string();
          if(topos.find(ghost_topo) != topos.end())
          {
            out_dom["fields/"+ghost].set_external(ghost_field);
          }
        }
      }
    }

    if(dom.has_path("state"))
    {
      out_dom["state"].set_external(dom["state"]);
    }
  }

  const int num_out_doms = output.number_of_children();
  bool has_data = false;
  // check to see if this resulted in any data
  for(int d = 0; d < num_out_doms; ++d)
  {
    const conduit::Node &dom = output.child(d);
    if(dom.has_path("fields"))
    {
      int fsize = dom["fields"].number_of_children();
      if(fsize != 0)
      {
        has_data = true;
        break;
      }
    }
  }

  has_data = global_someone_agrees(has_data);
  if(!has_data)
  {
    ASCENT_ERROR("Relay: field selection resulted in no data."
                 "This can occur if the fields did not exist "
                 "in the simulation data or if the fields were "
                 "created as a result of a pipeline, but the "
                 "relay extract did not receive the result of "
                 "a pipeline");
  }

}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// helper shared by io save and load
//-----------------------------------------------------------------------------
bool
verify_io_params(const conduit::Node &params,
                 conduit::Node &info)
{
    bool res = true;

    if( !params.has_child("path") )
    {
        info["errors"].append() = "missing required entry 'path'";
        res = false;
    }
    else if(!params["path"].dtype().is_string())
    {
        info["errors"].append() = "'path' must be a string";
        res = false;
    }
    else if(params["path"].as_string().empty())
    {
        info["errors"].append() = "'path' is an empty string";
        res = false;
    }


    if( params.has_child("protocol") )
    {
        if(!params["protocol"].dtype().is_string())
        {
            info["errors"].append() = "optional entry 'protocol' must be a string";
            res = false;
        }
        else if(params["protocol"].as_string().empty())
        {
            info["errors"].append() = "'protocol' is an empty string";
            res = false;
        }
        else
        {
            info["info"].append() = "includes 'protocol'";
        }
    }

    if( params.has_child("num_files") )
    {
        if(!params["num_files"].dtype().is_integer())
        {
            info["errors"].append() = "optional entry 'num_files' must be an integer";
            res = false;
        }
        else
        {
            info["info"].append() = "includes 'num_files'";
        }
    }

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("path");
    valid_paths.push_back("protocol");
    valid_paths.push_back("fields");
    valid_paths.push_back("num_files");
    ignore_paths.push_back("fields");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}


//-----------------------------------------------------------------------------
void mesh_blueprint_save(const Node &data,
                         const std::string &path,
                         const std::string &file_protocol,
                         int num_files,
                         std::string &root_file_out)
{
    bool has_data = blueprint::mesh::number_of_domains(data) > 0;
    has_data = global_someone_agrees(has_data);

    if(!has_data)
    {
      ASCENT_INFO("Blueprint save: no valid data exists. Skipping save");
      return;
    }

    // setup our options
    Node opts;
    opts["number_of_files"] = num_files;

#ifdef ASCENT_MPI_ENABLED
    conduit::relay::mpi::io::blueprint::save_mesh(data,
                                                  path,
                                                  file_protocol,
                                                  opts,
                                                  mpi_comm);
#else
    conduit::relay::io::blueprint::save_mesh(data,
                                             path,
                                             file_protocol,
                                             opts);
#endif

    return;

}



//-----------------------------------------------------------------------------
RelayIOSave::RelayIOSave()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOSave::~RelayIOSave()
{
// empty
}

//-----------------------------------------------------------------------------
void
RelayIOSave::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_save";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RelayIOSave::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void
RelayIOSave::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    path = output_dir(path);

    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    if(!input("in").check_type<DataObject>())
    {
        // error
        ASCENT_ERROR("relay_io_save requires a DataObject input");
    }

    DataObject *data_object  = input<DataObject>("in");
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<Node> n_input = data_object->as_node();

    Node *in = n_input.get();

    Node selected;

    if(params().has_path("fields"))
    {
      std::vector<std::string> field_selection;
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("relay_io_save field selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("relay_io_save field selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
      detail::filter_fields(*in, selected, field_selection, graph());
    }
    else
    {
      // select all fields
      selected.set_external(*in);
    }

    Node meta = Metadata::n_metadata;

    // Get the cycle and add it so filters don't have to
    // propagate this
    int cycle = -1;

    if(meta.has_path("cycle"))
    {
      cycle = meta["cycle"].as_int32();
    }
    if(cycle != -1)
    {
      const int num_domains = selected.number_of_children();
      for(int i = 0; i < num_domains; ++i)
      {
        // this could be external data, we don't want
        // to modify the source so if the entry exists
        // we remove and reset
        conduit::Node &dom = selected.child(i);
        if(dom.has_path("state/cycle"))
        {
            dom["state/cycle"].reset();
        }

        dom["state/cycle"] = cycle;
      }
    }

    int num_files = -1;

    if(params().has_path("num_files"))
    {
        num_files = params()["num_files"].to_int();
    }

    std::string result_path;
    if(protocol.empty())
    {
        conduit::relay::io::save(selected,path);
        result_path = path;
    }
    else if( protocol == "blueprint/mesh/hdf5" || protocol == "hdf5")
    {
        mesh_blueprint_save(selected,
                            path,
                            "hdf5",
                            num_files,
                            result_path);
    }
    else if( protocol == "blueprint/mesh/json" || protocol == "json")
    {
        mesh_blueprint_save(selected,
                            path,
                            "json",
                            num_files,
                            result_path);

    }
    else if( protocol == "blueprint/mesh/yaml" || protocol == "yaml")
    {
        mesh_blueprint_save(selected,
                            path,
                            "yaml",
                            num_files,
                            result_path);

    }
    else
    {
        conduit::relay::io::save(selected,path,protocol);
        result_path = path;
    }

    // add this to the extract results in the registry
    if(!graph().workspace().registry().has_entry("extract_list"))
    {
      conduit::Node *extract_list = new conduit::Node();
      graph().workspace().registry().add<Node>("extract_list",
                                               extract_list,
                                               -1); // TODO keep forever?
    }

    conduit::Node *extract_list = graph().workspace().registry().fetch<Node>("extract_list");

    Node &einfo = extract_list->append();
    einfo["type"] = "relay";
    if(!protocol.empty())
        einfo["protocol"] = protocol;
    einfo["path"] = result_path;
}



//-----------------------------------------------------------------------------
RelayIOLoad::RelayIOLoad()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOLoad::~RelayIOLoad()
{
// empty
}

//-----------------------------------------------------------------------------
void
RelayIOLoad::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_load";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
RelayIOLoad::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void
RelayIOLoad::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();

    // TODO check if we need to expand the path (MPI) case
    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    Node *res = new Node();

    if(protocol.empty())
    {
        conduit::relay::io::load(path,*res);
    }
    else
    {
        conduit::relay::io::load(path,protocol,*res);
    }

    set_output<Node>(res);

}
//-----------------------------------------------------------------------------
BlueprintFlatten::BlueprintFlatten()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintFlatten::~BlueprintFlatten()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintFlatten::declare_interface(Node &i)
{
    i["type_name"]   = "false";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
BlueprintFlatten::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("path") ||
       ! params["path"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'path'";
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("path");
    valid_paths.push_back("protocol");
    valid_paths.push_back("fields");

    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
        res = false;
	info["error"].append() = surprises;
    }

    return res;
    //return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void
BlueprintFlatten::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    path = output_dir(path);

    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    if(!input("in").check_type<DataObject>())
    {
        // error
        ASCENT_ERROR("Blueprint flatten requires a DataObject input");
    }

    DataObject *data_object  = input<DataObject>("in");
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<Node> n_input = data_object->as_node();

    Node *in = n_input.get();
    Node selected;
    if(params().has_path("fields"))
    {
      std::vector<std::string> field_selection;
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("Blueprint flatten field selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("Blueprint flatten field selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
      detail::filter_fields(*in, selected, field_selection, graph());
    }
    else
    {
      // select all fields
      selected.set_external(*in);
    }
    Node output;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    blueprint::mpi::mesh::flatten(selected,
		                  params(),
				  output,
  				  mpi_comm);
#else
    blueprint::mesh::flatten(selected,
		             params(),
			     output);

#endif

    std::string result_path;
    int rank = 0;
    int root = 0;
 
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm_rank(mpi_comm, &rank);
#endif

    if(rank == root)
    {
        if(protocol.empty())
        {
            //path = path;
            path = path + ".csv";
            conduit::relay::io::save(output,path);
            result_path = path;
        }
        else
        {
            conduit::relay::io::save(output,path,protocol);
	    result_path = path;
        }
    }

    // add this to the extract results in the registry
    if(!graph().workspace().registry().has_entry("extract_list"))
    {
      conduit::Node *extract_list = new conduit::Node();
      graph().workspace().registry().add<Node>("extract_list",
                                               extract_list,
                                               -1); // TODO keep forever?
    }

    conduit::Node *extract_list = graph().workspace().registry().fetch<Node>("extract_list");

    Node &einfo = extract_list->append();
    einfo["type"] = "flatten";
    if(!protocol.empty())
        einfo["protocol"] = protocol;
    einfo["path"] = result_path;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





