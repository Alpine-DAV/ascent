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

#if defined(ASCENT_VISKORES_ENABLED)
#include <viskores/cont/DataSet.h>
#include <ascent_vtkh_data_adapter.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#if defined(ASCENT_VISKORES_ENABLED)
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";

    string_schema(param_schema["properties/protocol"]);
    param_schema["required"].append() = "protocol";
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    number_schema(param_schema["properties/target"]);
    array_schema(param_schema["properties/fields"]);
    number_schema(param_schema["properties/mapping"]);
    number_schema(param_schema["properties/merge_tolerance"]);
    number_schema(param_schema["properties/build_adjsets"]);
    string_schema(param_schema["properties/original_element_ids"]);
    string_schema(param_schema["properties/original_vertex_ids"]);
    ignore_schema(param_schema["properties/distributed"]);

    // --- selections ---
    conduit::Node &selections_schema = param_schema["properties/selections"];
    selections_schema["type"] = "object";
    selections_schema["additionalProperties"] = false;
    string_schema(selections_schema["properties/type"]);
    string_schema(selections_schema["properties/topology"]);
    ignore_schema(selections_schema["properties/field"]);
    ignore_schema(selections_schema["properties/start"]);
    ignore_schema(selections_schema["properties/end"]);
    ignore_schema(selections_schema["properties/elements"]);
    ignore_schema(selections_schema["properties/ranges"]);
    selections_schema["required"].append() = "type";
    
    conduit::Node domain_id_schema = selections_schema["properties/domain_id"];
    domain_id_schema["type"] = "object";
    string_schema(domain_id_schema["oneOf"].append());
    number_schema(domain_id_schema["oneOf"].append());
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;
    param_schema["constraints/exclusiveChildren"].append() = "reduction_field";
    param_schema["constraints/exclusiveChildren"].append() = "var";
    param_schema["constraints/allowNoneInExclusiveGroup"] = false;

    ignore_schema(param_schema["properties/reduction_op"]);
    ignore_schema(param_schema["properties/reduction_field"]);
    ignore_schema(param_schema["properties/empty_bin_val"]);
    string_schema(param_schema["properties/output_type"]);
    ignore_schema(param_schema["properties/output_field"]);
    ignore_schema(param_schema["properties/var"]);

    // --- Axes ---
    {
        conduit::Node single_axis_schema;
        single_axis_schema["type"] = "object";
        single_axis_schema["additionalProperties"] = false;
        single_axis_schema["constraints/exclusiveChildren"].append() = "field";
        single_axis_schema["constraints/exclusiveChildren"].append() = "var";
        single_axis_schema["constraints/allowNoneInExclusiveGroup"] = false;

        number_schema(single_axis_schema["properties/min_val"], true);
        number_schema(single_axis_schema["properties/max_val"], true);
        number_schema(single_axis_schema["properties/num_bins"], true);
        number_schema(single_axis_schema["properties/clamp"], true);
        number_schema(single_axis_schema["properties/field"], true);
        number_schema(single_axis_schema["properties/var"], true);
        single_axis_schema["required"].append() = "num_bins";

        conduit::Node axes_schema = array_schema(param_schema["properties/axes"], single_axis_schema, 1, 3);
    }

    param_schema["required"].append() = "reduction_op";
    param_schema["required"].append() = "output_field";
    param_schema["required"].append() = "axes";
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/output_field"]);
    array_schema(param_schema["properties/fields"]);

    param_schema["required"].append() = "output_field";
    param_schema["required"].append() = "fields";
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/output_field"]);
    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/exponent"]);

    param_schema["required"].append() = "output_field";
    param_schema["required"].append() = "field";
    param_schema["required"].append() = "exponent";
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





