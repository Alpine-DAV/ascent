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
    int local_verify_ok = 0;
    if(!n_input->dtype().is_empty())
    {
        if(!conduit::blueprint::verify(protocol,
                                       *n_input,
                                       v_info))
        {
            n_input->schema().print();
            v_info.print();
            ASCENT_ERROR("blueprint verify failed for protocol"
                          << protocol << std::endl
                          << "details:" << std::endl
                          << v_info.to_json());
        }
        else
        {
            local_verify_ok = 1;
        }
    }

    // make sure some MPI task actually had bp data
#ifdef ASCENT_MPI_ENABLED
    int global_verify_ok = 0;
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Allreduce((void *)(&local_verify_ok),
                (void *)(&global_verify_ok),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
    local_verify_ok = global_verify_ok;
#endif

    if(local_verify_ok == 0)
    {
        ASCENT_ERROR("blueprint verify failed: published data is empty");
    }

    set_output<DataObject>(d_input);
}

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

    if(! params.has_child("target") ||
       ! params["target"].dtype().is_int() )
    {
        info["errors"].append() = "Missing required int parameter 'target'";
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
    
    conduit::Node n_options = params();

#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    conduit::blueprint::mpi::mesh::partition(*n_input,
		    			     n_options,
					     *n_output,
					     mpi_comm);
#else
    conduit::blueprint::mesh::partition(*n_input,
		     		        n_options,
					*n_output);
#endif
    DataObject *d_output = new DataObject(n_output);
    set_output<DataObject>(d_output);
}
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

    if(!params.has_path("var"))
    {
      res = false;
      info["errors"].append() = "Missing 'var'";
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("reduction_op");
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
        info["errors"].append() = "Number of axes num be between 1 and 3";
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
          if(!axis.has_path("var"))
          {
            res = false;
            info["errors"].append() = "Axis missing 'var'";
          }
          std::vector<std::string> avalid_paths;
          avalid_paths.push_back("min_val");
          avalid_paths.push_back("max_val");
          avalid_paths.push_back("num_bins");
          avalid_paths.push_back("clamp");
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
    std::string var = params()["var"].as_string();
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
      std::string axis_name = "value/"+in_axis["var"].as_string()+"/";
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

    expressions::binning_interface(var,
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
  mesh_in["attrs/reduction_var/value"] = var;
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
  else
  {
    //we already checked so this should not happen
    ASCENT_ERROR("Should never happen");
  }

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





