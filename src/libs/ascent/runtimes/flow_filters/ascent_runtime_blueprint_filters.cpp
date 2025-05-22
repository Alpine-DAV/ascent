//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_blueprint_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_blueprint_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mesh.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_metadata.hpp>
#include <runtimes/ascent_data_object.hpp>
#include <ascent_runtime_param_check.hpp>
#include "expressions/ascent_expression_filters.hpp"
#include "expressions/ascent_blueprint_architect.hpp"
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_blueprint_mpi_mesh.hpp>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/DataSet.h>
#include <ascent_vtkh_data_adapter.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/DataSet.hpp>
#endif

using namespace conduit;
using namespace std;

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
//-----------------------------------------------------------------------------
// BlueprintVerify
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
BlueprintVerify::BlueprintVerify()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintVerify::~BlueprintVerify()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintVerify::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_verify";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BlueprintVerify::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("protocol") ||
       ! params["protocol"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'protocol'";
    }

    return res;
}


//-----------------------------------------------------------------------------
void
BlueprintVerify::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_verify input must be a DataObject");
    }

    std::string protocol = params()["protocol"].as_string();

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    // some MPI tasks may not have data, that is fine
    // but blueprint verify will fail, so if the
    // input node is empty skip verify
    int local_verify_ok  = 0;
    int local_verify_err = 0;
    
    std::string verify_err_msg = "";
    if(!n_input->dtype().is_empty())
    {
        if(!conduit::blueprint::verify(protocol,
                                       *n_input,
                                       v_info))
        {
            verify_err_msg = v_info.to_yaml();
            local_verify_err = 1;
        }
        else
        {
            local_verify_ok = 1;
        }
    }

    // make sure some MPI task actually had bp data
#ifdef ASCENT_MPI_ENABLED
    // reduce flag for some valid data
    int global_verify_ok = 0;
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Allreduce((void *)(&local_verify_ok),
                  (void *)(&global_verify_ok),
                  1,
                  MPI_INT,
                  MPI_SUM,
                  mpi_comm);
    local_verify_ok = global_verify_ok;

    // reduce flag for errors
    int global_verify_err = 0;
    MPI_Allreduce((void *)(&local_verify_err),
                  (void *)(&global_verify_err),
                  1,
                  MPI_INT,
                  MPI_SUM,
                  mpi_comm);
    local_verify_err = global_verify_err;


#endif

    // check for an error on any rank
    if(local_verify_err == 1)
    {
        if(verify_err_msg != "")
        {
            ASCENT_ERROR("blueprint verify failed for protocol"
                          << protocol << std::endl
                          << "one one more more ranks." << std::endl
                          << "Details:" << std::endl
                          << verify_err_msg);
        } 
        else
        {
            ASCENT_ERROR("blueprint verify failed for protocol"
                          << protocol << std::endl
                          << "one one more more ranks." << std::endl);
        }
    }

    // check for no data
    if(local_verify_ok == 0)
    {
        ASCENT_ERROR("blueprint verify failed: published data is empty");
    }

    set_output<DataObject>(d_input);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// ConduitExtract
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
ConduitExtract::ConduitExtract()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ConduitExtract::~ConduitExtract()
{
// empty
}

//-----------------------------------------------------------------------------
void
ConduitExtract::declare_interface(Node &i)
{
    i["type_name"]   = "conduit_extract";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ConduitExtract::verify_params(const conduit::Node &params,
                              conduit::Node &info)
{
    info.reset();
    bool res = true;

    // so far, no params

    return res;
}

//-----------------------------------------------------------------------------
void
ConduitExtract::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("conduit_extract input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    // squirrel a copy away in the registry where it will
    // be connected with exec info

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
    einfo["type"] = "conduit";
    einfo["data"].set(*n_input);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// BlueprintPartition
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
BlueprintPartition::BlueprintPartition()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintPartition::~BlueprintPartition()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintPartition::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_data_partition";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BlueprintPartition::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    res &= check_numeric("target",params, info, false, false);

    if(params.has_child("selections"))
    {
      res &= check_string("selections/type",params, info, true);
      //domain_id can be int or "any"
      res &= (check_string("selections/domain_id",params, info, false) || check_numeric("selections/domain_id", params, info, false, false));
      res &= check_string("selections/topology",params, info, false);
    }

    if(params.has_child("fields"))
    {
      if(!params["fields"].dtype().is_list())
      {
        res = false;
        info["errors"].append() = "fields is not a list";
      }
    }

    res &= check_numeric("mapping",params, info, false, false);
    res &= check_numeric("merge_tolerance",params, info, false, false);
    res &= check_numeric("build_adjsets",params, info, false, false);
    res &= check_string("original_element_ids",params, info, false);
    res &= check_string("original_vertex_ids",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("target");
    valid_paths.push_back("selections/type");
    valid_paths.push_back("selections/domain_id");
    valid_paths.push_back("selections/field");
    valid_paths.push_back("selections/topology");
    valid_paths.push_back("selections/start");
    valid_paths.push_back("selections/end");
    valid_paths.push_back("selections/elements");
    valid_paths.push_back("selections/ranges");
    valid_paths.push_back("selections/field");
    valid_paths.push_back("mapping");
    valid_paths.push_back("merge_tolerance");
    valid_paths.push_back("build_adjsets");
    valid_paths.push_back("original_element_ids");
    valid_paths.push_back("original_vertex_ids");
    valid_paths.push_back("distributed");

    std::vector<std::string> ingore_paths = {"fields"};

    std::string surprises = surprise_check(valid_paths,ingore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}


//-----------------------------------------------------------------------------
void
BlueprintPartition::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_data_partition input must be a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    conduit::Node *n_output = new conduit::Node();
    
    conduit::Node n_options;

    int target = 1;
    if(params().has_child("target"))
    {
      target = params()["target"].to_int32();
    }

    n_options.set_external(params());
    if(n_options.has_child("distributed"))
    {
      n_options.remove_child("distributed");
    }

    conduit::Node tmp;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    if(params().has_child("distributed") && 
       params()["distributed"].as_string() == "false" )
    {
        conduit::blueprint::mesh::partition(*n_input,
                                            n_options,
                                            tmp);
    }
    else
    {
        conduit::blueprint::mpi::mesh::partition(*n_input,
                                                 n_options,
                                                 tmp,
                                                 mpi_comm);
    }
#else
    conduit::blueprint::mesh::partition(*n_input,
                                        n_options,
                                        tmp);
#endif

    if(tmp.number_of_children() > 0)
    {
      if(target == 1)
      {
        n_output->append().move(tmp);
      }
      else
      {
        n_output->move(tmp);
      }
    }
    DataObject *d_output = new DataObject(n_output);
    set_output<DataObject>(d_output);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// DataBinning
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
DataBinning::DataBinning()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DataBinning::~DataBinning()
{
// empty
}

//-----------------------------------------------------------------------------
void
DataBinning::declare_interface(Node &i)
{
    i["type_name"]   = "data_binning";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DataBinning::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(!params.has_path("reduction_op"))
    {
      res = false;
      info["errors"].append() = "Missing 'reduction_op'";
    }

    // note: `var` is deprecated, new arg reduction_field
    if(!params.has_path("reduction_field"))
    {
      if(!params.has_path("var"))
      {
        res = false;
        info["errors"].append() = "Missing 'reduction_field'";
      }
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("reduction_op");
    valid_paths.push_back("reduction_field");
    valid_paths.push_back("empty_bin_val");
    valid_paths.push_back("output_type");
    valid_paths.push_back("output_field");
    valid_paths.push_back("var");

    std::vector<std::string> ignore_paths;
    ignore_paths.push_back("axes");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(!params.has_path("output_field"))
    {
      res = false;
      info["errors"].append() = "Missing param 'output_field'";
    }

    if(!params.has_path("axes"))
    {
      res = false;
      info["errors"].append() = "Missing binning axes";
    }
    else if(!params["axes"].dtype().is_list())
    {
      res = false;
      info["errors"].append() = "Axes is not a list";
    }
    else
    {
      const int num_axes = params["axes"].number_of_children();
      if(num_axes < 1 || num_axes > 3)
      {
        res = false;
        info["errors"].append() = "Number of axes must be between 1 and 3";
      }
      else
      {
        for(int i = 0; i < num_axes; ++i)
        {
          const conduit::Node &axis = params["axes"].child(i);
          if(!axis.has_path("num_bins"))
          {
            res = false;
            info["errors"].append() = "Axis missing 'num_bins'";
          }
          
          if(!axis.has_child("field"))
          {
            if(!axis.has_child("var"))
            {
              std::ostringstream oss;
              oss << "Axis " << i << " missing 'field' parameter";
              res = false;
              info["errors"].append() = oss.str();
            }
          }

          std::vector<std::string> avalid_paths;
          avalid_paths.push_back("min_val");
          avalid_paths.push_back("max_val");
          avalid_paths.push_back("num_bins");
          avalid_paths.push_back("clamp");
          avalid_paths.push_back("field");
          avalid_paths.push_back("var");

          surprises += surprise_check(avalid_paths, axis);
        }
      }
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}


//-----------------------------------------------------------------------------
void
DataBinning::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("data binning input must be a DataObject");
    }

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_low_order_bp();

    std::string reduction_op = params()["reduction_op"].as_string();
    std::string reduction_field;

    // `var` is deprecated, new style arg: `reduction_field`
    if(params().has_child("reduction_field"))
    {
      reduction_field = params()["reduction_field"].as_string();
    }
    else if(params().has_child("var"))
    {
      reduction_field = params()["var"].as_string();
    }
    else
    {
        ASCENT_ERROR("Data Binning: Missing `reduction_field` parameter");
    }
    conduit::Node n_component;

    std::string output_type = "mesh";

    if(params().has_path("output_type"))
    {
      output_type = params()["output_type"].as_string();
      if(output_type != "mesh" && output_type != "bins")
      {
        ASCENT_ERROR("output type can only be 'mesh' or 'bins'");
      }
    }

    std::string output_field = params()["output_field"].as_string();

    if(params().has_path("component"))
    {
      n_component = params()["component"];
    }

    conduit::Node n_empty_bin_val;
    if(params().has_path("empty_bin_val"))
    {
      n_empty_bin_val = params()["empty_bin_val"];
    }

    conduit::Node n_axes_list;
    n_axes_list["type"] = "list";
    conduit::Node &n_axes = n_axes_list["value"];
    const int num_axes = params()["axes"].number_of_children();
    for(int i = 0; i < num_axes; ++i)
    {
      const conduit::Node &in_axis = params()["axes"].child(i);
      // transform into a for that expressions wants
      conduit::Node &axis = n_axes.append();
      
      std::string axis_field_name;

      if(in_axis.has_path("field"))
      {
        axis_field_name = in_axis["field"].as_string();
      }
      else if(in_axis.has_path("var"))
      {
        axis_field_name = in_axis["var"].as_string();
      }
      else
      {
          ASCENT_ERROR("Data Binning: axis " << i <<
                       " is missing `field` parameter");
      }

      std::string axis_name = "value/" + axis_field_name + "/";
      axis["type"] = "axis";
      axis[axis_name+"num_bins"] = in_axis["num_bins"];
      if(in_axis.has_path("min_val"))
      {
        axis[axis_name+"min_val"] = in_axis["min_val"];
      }
      if(in_axis.has_path("max_val"))
      {
        axis[axis_name+"max_val"] = in_axis["max_val"];
      }
      int clamp = 0;
      if(in_axis.has_path("clamp"))
      {
        clamp = in_axis["clamp"].to_int32();
      }
      axis[axis_name+"clamp"] = clamp;

    }

    conduit::Node n_binning;
    conduit::Node n_output_axes;

    expressions::binning_interface(reduction_field,
                                   reduction_op,
                                   n_empty_bin_val,
                                   n_component,
                                   n_axes_list,
                                   *n_input.get(),
                                   n_binning,
                                   n_output_axes);

  // setup the input to the painting functions
  conduit::Node mesh_in;
  mesh_in["type"] = "binning";
  mesh_in["attrs/value/value"] = n_binning["value"];
  mesh_in["attrs/value/type"] = "array";
  // TODO: Re plumb binning mesh args
  mesh_in["attrs/reduction_var/value"] = reduction_field;
  mesh_in["attrs/reduction_var/type"] = "string";
  mesh_in["attrs/reduction_op/value"] = reduction_op;
  mesh_in["attrs/reduction_op/type"] = "string";
  mesh_in["attrs/bin_axes/value"] = n_output_axes;
  mesh_in["attrs/association/value"] = n_binning["association"];
  mesh_in["attrs/association/type"] = "string";

  if(output_type == "bins")
  {
    Node meta = Metadata::n_metadata;
    int cycle = -1;
    double time = -1.0;
    if(meta.has_path("cycle"))
    {
      cycle = meta["cycle"].to_int32();
    }
    if(meta.has_path("time"))
    {
      time = meta["time"].to_float64();
    }
    // create a new reduced size mesh from the binning
    conduit::Node *out_data = new conduit::Node();
    // we only have one data set so give this to rank 0

    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    int comm_id = flow::Workspace::default_mpi_comm();
    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
    MPI_Comm_rank(mpi_comm,&rank);
#endif

    if(rank == 0)
    {
      conduit::Node &n_binning_mesh = out_data->append();
      expressions::binning_mesh(mesh_in, n_binning_mesh, output_field);
      n_binning_mesh["state/cycle"] = cycle;
      n_binning_mesh["state/time"] = time;
      n_binning_mesh["state/domain_id"] = 0;
    }

    DataObject  *d_output = new DataObject();
    d_output->reset(out_data);
    d_output->name("binning");
    set_output<DataObject>(d_output);
  }
  else if(output_type== "mesh")
  {
    // we are taking the shared pointer from the input so
    // we don't copy anything extra
    DataObject  *d_output = new DataObject();
    d_output->reset(n_input);
    expressions::paint_binning(mesh_in, *n_input.get(), output_field);
    set_output<DataObject>(d_output);
  }
  else if(output_type== "samples")
  {
    // create a point mesh that has the sample points and value

    DataObject  *d_output = new DataObject();
    d_output->reset(n_input);
    expressions::paint_binning(mesh_in, *n_input.get());
    set_output<DataObject>(d_output);

    // // we are taking the shared pointer from the input so
    // // we don't copy anything extra
    // DataObject  *d_output = new DataObject();
    // d_output->reset(n_input);
    // expressions::paint_binning(mesh_in, *n_input.get(), output_field);
    // set_output<DataObject>(d_output);
  }
  else
  {
    //we already checked so this should not happen
    ASCENT_ERROR("Should never happen");
  }

}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// AddFields (derived field)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
AddFields::AddFields() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
AddFields::~AddFields()
{
  // empty
}
//-----------------------------------------------------------------------------
void
AddFields::declare_interface(Node &i)
{
    i["type_name"]   = "add_fields";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
AddFields::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;


    if(!params.has_path("output_field"))
    {
      res = false;
      info["errors"].append() = "Missing param 'output_field'";
    }

    if(!params.has_path("fields"))
    {
      res = false;
      info["errors"].append() = "Missing 'fields'";
    }
    else if(!params["fields"].dtype().is_list())
    {
      res = false;
      info["errors"].append() = "fields is not a list";
    }

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("fields");
    valid_paths.push_back("output_field");
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
void
AddFields::execute()
{
  if(!input(0).check_type<DataObject>())
  {
      ASCENT_ERROR("add fields input must be a DataObject");
  }

  Node v_info;
  DataObject *d_input = input<DataObject>(0);
  std::shared_ptr<conduit::Node> n_input = d_input->as_low_order_bp();

  std::string out_field = params()["output_field"].as_string();
  std::vector<std::string> fields;
  const conduit::Node &flist = params()["fields"];
  const int num_fields = flist.number_of_children();
  if(num_fields == 0)
  {
    ASCENT_ERROR("'fields' list must be non-empty");
  }
  for(int i = 0; i < num_fields; i++)
  {
    const conduit::Node &f = flist.child(i); 
    if(!f.dtype().is_string())
    {
      ASCENT_ERROR("'fields' list values must be a string");
    }
    fields.push_back(f.as_string());
  }

  DataObject  *d_output = new DataObject();
  d_output->reset(n_input);
  expressions::derived_field_add_fields(*n_input.get(), fields, out_field);
  set_output<DataObject>(d_output);

}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// PowerOfField (derived field)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
PowerOfField::PowerOfField() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
PowerOfField::~PowerOfField()
{
  // empty
}
//-----------------------------------------------------------------------------
void
PowerOfField::declare_interface(Node &i)
{
    i["type_name"]   = "power_of_field";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
PowerOfField::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;


    if(!params.has_path("output_field"))
    {
      res = false;
      info["errors"].append() = "Missing param 'output_field'";
    }

    if(!params.has_path("field"))
    {
      res = false;
      info["errors"].append() = "Missing param 'field'";
    }

    if(!params.has_path("exponent"))
    {
      res = false;
      info["errors"].append() = "Missing param 'exponent'";
    }

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("exponent");
    valid_paths.push_back("output_field");


    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
PowerOfField::execute()
{
  if(!input(0).check_type<DataObject>())
  {
      ASCENT_ERROR("add fields input must be a DataObject");
  }

  Node v_info;
  DataObject *d_input = input<DataObject>(0);
  std::shared_ptr<conduit::Node> n_input = d_input->as_low_order_bp();

  std::string out_field = params()["output_field"].as_string();
  std::string field = params()["field"].as_string();
  double exponent = 0.0; 
  if(params()["exponent"].dtype().is_int())
  {
    exponent = (double)params()["exponent"].as_int(); 	  
  }
  else if(params()["exponent"].dtype().is_float32())
  {
    exponent = (double)params()["exponent"].as_float32(); 	  
  }
  else if(params()["exponent"].dtype().is_float64())
  {
    exponent = params()["exponent"].as_float64(); 	  
  }
  else
    ASCENT_ERROR("'exponent' type not recognized, must be a number");

  DataObject  *d_output = new DataObject();
  d_output->reset(n_input);
  expressions::derived_field_power_of_field(*n_input.get(), field, exponent, out_field);
  set_output<DataObject>(d_output);

}

//-----------------------------------------------------------------------------
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





