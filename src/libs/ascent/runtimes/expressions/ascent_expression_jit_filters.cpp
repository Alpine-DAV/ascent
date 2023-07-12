//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ascent_expression_jit_filters.hpp"
#include "ascent_jit_fusion.hpp"
#include "ascent_blueprint_architect.hpp"
#include "ascent_blueprint_topologies.hpp"
#include <ascent_config.h>
#include <ascent_logging.hpp>
#include <ascent_data_object.hpp>
#include <ascent_runtime_param_check.hpp>
#include <utils/ascent_mpi_utils.hpp>
#include <flow_graph.hpp>
#include <flow_timer.hpp>
#include <flow_workspace.hpp>

#include <list>

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
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{
    
//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------
namespace detail
{

//-----------------------------------------------------------------------------
std::string
fused_kernel_type(const std::vector<std::string> kernel_types)
{
  std::set<std::string> topo_types;
  for(const auto &kernel_type : kernel_types)
  {
    size_t last = 0;
    size_t next = 0;
    while((next = kernel_type.find(";", last)) != string::npos)
    {
      topo_types.insert(kernel_type.substr(last, next - last));
      last = next + 1;
    }
    topo_types.insert(kernel_type.substr(last));
  }

  topo_types.erase("default");
  if(topo_types.empty())
  {
    return "default";
  }

  std::stringstream ss;
  bool first = true;
  for(const auto &topo_type : topo_types)
  {
    if(!first)
    {
      ss << ";";
    }
    ss << topo_type;
    first = false;
  }
  return ss.str();
}

//-----------------------------------------------------------------------------
void
topo_to_jitable(const std::string &topology,
                const conduit::Node &dataset,
                Jitable &jitable)
{
  for(int i = 0; i < dataset.number_of_children(); ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    std::unique_ptr<Topology> topo = topologyFactory(topology, dom);
    pack_topology(
        topology, dom, jitable.dom_info.child(i)["args"], jitable.arrays[i]);
    const std::string kernel_type = topology + "=" + topo->topo_type;
    jitable.dom_info.child(i)["kernel_type"] = kernel_type;
    jitable.kernels[kernel_type];
  }
  jitable.topology = topology;
}
}
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------

//******************************************************************************
// ExprJitFilter
//******************************************************************************

//-----------------------------------------------------------------------------
ExprJitFilter::ExprJitFilter(
    const int num_inputs,
    const std::shared_ptr<const JitExecutionPolicy> exec_policy)
: Filter(),
  num_inputs(num_inputs),
  exec_policy(exec_policy)
{
}

//-----------------------------------------------------------------------------
ExprJitFilter::~ExprJitFilter()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ExprJitFilter::declare_interface(Node &i)
{
  stringstream ss;
  ss << "expr_jit_filter_" << num_inputs << "_" << exec_policy->get_name();
  i["type_name"] = ss.str();
  for(int inp_num = 0; inp_num < num_inputs; ++inp_num)
  {
    std::stringstream ss;
    ss << "arg" << inp_num;
    i["port_names"].append() = ss.str();
  }
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ExprJitFilter::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
  info.reset();
  bool res = filters::check_string("func", params, info, true);
  res &= filters::check_string("filter_name", params, info, true);
  if(!params.has_path("inputs"))
  {
    info["errors"].append() = "Missing required parameter 'inputs'";
    res = false;
  }
  else if(params["inputs"].number_of_children() != num_inputs)
  {
    stringstream ss;
    ss << "Expected parameter 'inputs' to have " << num_inputs
       << " inputs but it has " << params["inputs"].number_of_children()
       << " inputs.";
    info["errors"].append() = ss.str();
    res = false;
  }
  return res;
}

//-----------------------------------------------------------------------------
// each jitable has kernels and dom_info
// dom_info holds number of entries, kernel_type, and args for the dom
// kernel_type maps to a kernel in kernels
// each kernel has 3 bodies of code:
// expr: the main expression being transformed (e.g topo_volume * density[item])
// for_body: for-loop body that holds code needed for expr
// kernel_body: code that we already generated but aren't touching (i.e. past
// for-loops). for_body gets wrapped in a for-loop and added to kernel_body when
// we need to generate an temporary field.
//-----------------------------------------------------------------------------
void
ExprJitFilter::execute()
{
  // There are a lot of possible code paths that each rank/domain could
  // follow. All JIT code should not contain any MPI calls, but things
  // after JIT can call MPI, so its important that we globally catch errors
  // and halt execution on any error. Otherwise, we will deadlock
  conduit::Node errors;
  try
  {
    const std::string &func = params()["func"].as_string();
    const std::string &filter_name = params()["filter_name"].as_string();
    const conduit::Node &inputs = params()["inputs"];

    // don't execute if we just executed
    if(func == "execute" && input(0).check_type<conduit::Node>())
    {
      set_output(input<conduit::Node>(0));
      return;
    }

    // create a vector of input_jitables to be fused
    std::vector<const Jitable *> input_jitables;
    // keep around the new jitables we create
    std::list<Jitable> new_jitables;

    DataObject *data_object =
      graph().workspace().registry().fetch<DataObject>("dataset");
    conduit::Node *dataset = data_object->as_low_order_bp().get();
    const int num_domains = dataset->number_of_children();

    // registry node that stores temporary arrays
    conduit::Node *const remove =
        graph().workspace().registry().fetch<Node>("remove");

    // convert filter's inputs (numbers, topos, fields, binnings, etc.) to jitables
    for(int i = 0; i < num_inputs; ++i)
    {
      const std::string input_fname = inputs.child(i)["filter_name"].as_string();
      const std::string type = inputs.child(i)["type"].as_string();
      // A jitable at "compile time" may have been executed at runtime so we
      // need check_type to get the runtime type to make sure it hasn't executed
      if(type == "jitable" && input(i).check_type<Jitable>())
      {
        // push back an existing jitable
        input_jitables.push_back(input<Jitable>(i));
      }
      else
      {
        const conduit::Node *inp = input<conduit::Node>(i);
        // make a new jitable
        new_jitables.emplace_back(num_domains);
        Jitable &jitable = new_jitables.back();
        input_jitables.push_back(&jitable);

        if(type == "topology")
        {
          // topo is special because it can build different kernels for each
          // domain (kernel types)
          detail::topo_to_jitable((*inp)["value"].as_string(),
                                  *dataset,
                                  jitable);
          jitable.obj = *inp;
        }
        else
        {
          // default kernel type means we don't need to generate any
          // topology-specific code.
          // During kernel fusion the type of the output kernel is the "union" of
          // the types of the input kernels.
          Kernel &default_kernel = jitable.kernels["default"];
          for(int i = 0; i < num_domains; ++i)
          {
            jitable.dom_info.child(i)["kernel_type"] = "default";
          }

          if(type == "int" || type == "double" || type == "bool")
          {
            // force everything to a double
            for(int i = 0; i < num_domains; ++i)
            {
              jitable.dom_info.child(i)["args/" + input_fname] = (*inp)["value"];
            }
            default_kernel.expr = "((double)" + input_fname + ")";
            default_kernel.num_components = 1;
          }
          else if(type == "vector")
          {
            for(int i = 0; i < num_domains; ++i)
            {
              jitable.dom_info.child(i)["args/" + input_fname] = (*inp)["value"];
            }
            default_kernel.expr = input_fname;
            default_kernel.num_components = 3;
          }
          // field or a jitable that was executed at runtime
          else if(type == "field" || type == "jitable")
          {
            std::string field_name = (*inp)["value"].as_string();
            // error checking and dom args information
            for(int i = 0; i < num_domains; ++i)
            {
              const conduit::Node &dom = dataset->child(i);
              const conduit::Node &field = dom["fields/" + field_name];
              const std::string &topo_name = field["topology"].as_string();
              const std::string &assoc_str = field["association"].as_string();

              conduit::Node &cur_dom_info = jitable.dom_info.child(i);

              std::string values_path = "values";
              if(inp->has_path("component"))
              {
                const std::string &component = (*inp)["component"].as_string();
                values_path += "/" + component;
                field_name += "_" + component;
                default_kernel.num_components = 1;
              }
              else
              {
                const int num_children = field[values_path].number_of_children();
                default_kernel.num_components = std::max(1, num_children);
              }

              bool is_float64;
              if(field[values_path].number_of_children() > 1)
              {
                is_float64 = field[values_path].child(0).dtype().is_float64();
              }
              else
              {
                is_float64 = field[values_path].dtype().is_float64();
              }

              pack_array(field[values_path],
                         field_name,
                         cur_dom_info["args"],
                         jitable.arrays[i]);

              // update number of entries
              int entries;
              std::unique_ptr<Topology> topo = topologyFactory(topo_name, dom);
              if(assoc_str == "element")
              {
                entries = topo->get_num_cells();
              }
              else
              {
                entries = topo->get_num_points();
              }
              cur_dom_info["entries"] = entries;

              // update topology
              if(!jitable.topology.empty())
              {
                if(jitable.topology != topo_name)
                {
                  ASCENT_ERROR("Field '" << field_name
                                         << "' is associated with different "
                                            "topologies on different domains.");
                }
              }
              else
              {
                jitable.topology = topo_name;
              }

              // update association
              if(!jitable.association.empty())
              {
                if(jitable.association != assoc_str)
                {
                  ASCENT_ERROR(
                      "Field '"
                      << field_name
                      << "' has different associations on different domains.");
                }
              }
              else
              {
                jitable.association = assoc_str;
              }

              // Used to determine if we need to generate an entire derived field
              // beforehand for things like gradient(field).
              // obj["value"] will have the name of the field, "value" goes away
              // as soon as we do something like field + 1, indicating we are no
              // longer dealing with an original field and will have to generate
              // it via a for-loop.
              jitable.obj["value"] = field_name;
              jitable.obj["type"] = "field";
            }
            // We assume that fields don't have different strides/offsets in
            // different domains (otherwise we might need to compile a kernel for
            // every domain) so just use the first domain array for codegen
            if(default_kernel.num_components == 1)
            {
              default_kernel.expr = jitable.arrays[0].index(field_name, "item");
            }
            else
            {
              default_kernel.for_body.insert(
                  "double " + field_name + "_item[" +
                  std::to_string(default_kernel.num_components) + "];\n");
              for(int i = 0; i < default_kernel.num_components; ++i)
              {
                default_kernel.for_body.insert(
                    field_name + "_item[" + std::to_string(i) + "] = " +
                    jitable.arrays[0].index(field_name, "item", i) + ";\n");
              }
              default_kernel.expr = field_name + "_item";
            }
          }
          else if(type == "binning")
          {
            // we need to put the binning in the registry, otherwise it may get
            // deleted
            conduit::Node &binning_value = (*remove)["temporaries"].append();
            binning_value = (*inp)["attrs/value/value"];
            for(int i = 0; i < num_domains; ++i)
            {
              conduit::Node &args = jitable.dom_info.child(i)["args"];
              // pack the binning array
              // TODO this is the same for every domain and it's getting copied...
              pack_array(
                  binning_value, input_fname + "_value", args, jitable.arrays[i]);

              // pack the axes
              // if axis is a field pack the field
              const conduit::Node &axes = (*inp)["attrs/bin_axes/value"];
              for(int i = 0; i < axes.number_of_children(); ++i)
              {
                const conduit::Node &axis = axes.child(i);
                const std::string &axis_name = axis.name();
                const std::string axis_prefix =
                    input_fname + "_" + axis_name + "_";
                if(axis.has_path("num_bins"))
                {
                  args[axis_prefix + "min_val"] = axis["min_val"];
                  args[axis_prefix + "max_val"] = axis["max_val"];
                  args[axis_prefix + "num_bins"] = axis["num_bins"];
                }
                else
                {
                  pack_array(args["bins"],
                             axis_prefix + "bins",
                             args,
                             jitable.arrays[i]);
                  args[axis_prefix + "bins_len"] =
                      axis["bins"].dtype().number_of_elements();
                }
                args[axis_prefix + "clamp"] = axis["clamp"];
                if(!is_xyz(axis_name))
                {
                  if(!has_field(*dataset, axis_name))
                  {
                    ASCENT_ERROR("Could not find field '"
                                 << axis_name
                                 << "' in the dataset while packing binning.");
                  }
                  const conduit::Node &dom = dataset->child(i);
                  const conduit::Node &values =
                      dom["fields/" + axis_name + "/values"];
                  pack_array(values, axis_name, args, jitable.arrays[i]);
                }
                // we may not need the topology associated with the binning if we
                // are painting to a different topology so don't pack it here
              }
            }
          }
          else if(type == "string")
          {
            // strings don't get converted to jitables, they are used as arguments
            // to jitable functions
            jitable.obj = *inp;
          }
          else
          {
            ASCENT_ERROR("Cannot convert object of type '" << type
                                                           << "' to jitable.");
          }
        }
      }
    }

    // fuse
    Jitable *out_jitable = new Jitable(num_domains);
    // fuse jitable variables (e.g. entries, topo, assoc) and args
    for(const Jitable *input_jitable : input_jitables)
    {
      out_jitable->fuse_vars(*input_jitable);
    }

    // some functions need to pack the topology but don't take it in as an
    // argument. hack: add a new input jitable to the end with the topology and
    // fuse it
    if(func == "expr_jit_mesh_field_gradient" ||
       func == "expr_jit_mesh_field_curl"     ||
       func == "expr_jit_mesh_field_recenter" ||
       (func == "expr_jit_mesh_binning_value" && !inputs.has_path("topo")))
    {
      new_jitables.emplace_back(num_domains);
      Jitable &jitable = new_jitables.back();
      input_jitables.push_back(&jitable);

      std::string topology;
      if(func == "expr_jit_mesh_binning_value")
      {
        // if a topology wasn't passed in get the one associated with the binning
        const int binning_port = inputs["binning/port"].to_int32();
        const conduit::Node &binning = *input<conduit::Node>(binning_port);
        topology = binning["attrs/topology/value"].as_string();
        if(!has_topology(*dataset, topology))
        {
          std::set<std::string> names = topology_names(*dataset);
          std::stringstream msg;
          msg<<"Unknown topology: '"<<topology<<"'. Known topologies: [";
          for(auto &name : names)
          {
            msg<<" "<<name;
          }
          msg<<" ]";
          ASCENT_ERROR(msg.str());
        }
      }
      else
      {
        topology = out_jitable->topology;
      }
      detail::topo_to_jitable(topology, *dataset, jitable);

      out_jitable->fuse_vars(jitable);
    }

    if(func == "execute")
    {
      // just copy over the existing kernels, no need to fuse
      out_jitable->kernels = input_jitables[0]->kernels;
    }
    else
    {
      // fuse kernels
      std::unordered_set<std::string> fused_kernel_types;
      for(int dom_idx = 0; dom_idx < num_domains; ++dom_idx)
      {
        // get the input kernels with the right kernel_type for this domain and
        // determine the type of the fused kernel
        std::vector<const Kernel *> input_kernels;
        std::vector<std::string> input_kernel_types;
        for(const Jitable *input_jitable : input_jitables)
        {
          const std::string kernel_type =
              input_jitable->dom_info.child(dom_idx)["kernel_type"].as_string();
          input_kernel_types.push_back(kernel_type);
          input_kernels.push_back(&(input_jitable->kernels.at(kernel_type)));
        }
        const std::string out_kernel_type = detail::fused_kernel_type(input_kernel_types);
        (*out_jitable).dom_info.child(dom_idx)["kernel_type"] = out_kernel_type;
        Kernel &out_kernel = out_jitable->kernels[out_kernel_type];
        const bool not_fused =
            fused_kernel_types.find(out_kernel_type) == fused_kernel_types.cend();

        // this class knows how to combine kernels and generate jitable functions
        JitableFusion jitable_functions(params(),
                                        input_jitables,
                                        input_kernels,
                                        filter_name,
                                        *dataset,
                                        dom_idx,
                                        not_fused,
                                        *out_jitable,
                                        out_kernel);

        if(func == "expr_jit_binary_op")
        {
          jitable_functions.binary_op();
        }
        // ---------------------------------------------------------------------
        // These cases are functions that can just be called in OCCA
        // Map filter_name from the function signature to function name in OCCA
        // ---------------------------------------------------------------------
        else if(func == "expr_jit_mesh_field_min")
        {
            jitable_functions.builtin_functions("min");
        }
        else if(func == "expr_jit_mesh_field_max")
        {
            jitable_functions.builtin_functions("max");
        }
        else if(func == "expr_jit_mesh_field_sin")
        {
            jitable_functions.builtin_functions("sin");
        }
        else if(func == "expr_jit_mesh_field_sqrt")
        {
            jitable_functions.builtin_functions("sqrt");
        }
        else if(func == "expr_jit_mesh_field_pow")
        {
            jitable_functions.builtin_functions("pow");
        }
        else if(func == "expr_jit_mesh_field_abs")
        {
            jitable_functions.builtin_functions("abs");
        }
        else if(func == "expr_jit_rand")
        {
          jitable_functions.rand();
        }
        else if(func == "expr_jit_dot")
        {
          jitable_functions.expr_dot();
        }
        else if(func == "expr_jit_if")
        {
          jitable_functions.expr_if();
        }
        else if(func == "expr_jit_mesh_field_constant")
        {
          jitable_functions.constant_field();
        }
        else if(func == "expr_jit_mesh_field_vector_compose")
        {
          jitable_functions.vector();
        }
        else if(func == "expr_jit_vector_magnitude")
        {
          jitable_functions.magnitude();
        }
        else if(func == "expr_jit_mesh_field_gradient")
        {
          jitable_functions.gradient();
        }
        else if(func == "expr_jit_mesh_field_curl")
        {
          jitable_functions.curl();
        }
        else if(func == "expr_jit_mesh_binning_paint_binning")
        {
          const int binning_port = inputs["binning/port"].to_int32();
          const conduit::Node &binning = *input<conduit::Node>(binning_port);
          jitable_functions.binning_value(binning);
        }
        else if(func == "expr_jit_mesh_field_recenter")
        {
          jitable_functions.recenter();
        }
        else
        {
          ASCENT_ERROR("ExprJitFilter: Unknown func: '" << func << "'");
        }
        fused_kernel_types.insert(out_kernel_type);
      }
    }

    if(exec_policy->should_execute(*out_jitable))
    {
      std::string field_name;
      if(params().has_path("field_name"))
      {
        // perm name for the new node
        field_name = params()["field_name"].as_string();
      }
      else
      {
        field_name = filter_name;
        // mark this for cleanup
        (*remove)["fields/" + filter_name];
      }

      out_jitable->execute(*dataset, field_name);

      Node *output = new conduit::Node();

      (*output)["value"] = field_name;
      (*output)["type"] = "field";
      set_output<conduit::Node>(output);

      delete out_jitable;
    }
    else
    {
      set_output<Jitable>(out_jitable);
    }
  } // try
  catch(conduit::Error &e)
  {
    errors.append() = e.what();
  }
  catch(std::exception &e)
  {
    errors.append() = e.what();
  }
  catch(...)
  {
    errors.append() = "Unknown error occurred in JIT";
  }

  bool error = errors.number_of_children() > 0;
  error = global_someone_agrees(error);
  if(error)
  {
    std::set<std::string> error_strs;
    for(int i = 0; i < errors.number_of_children(); ++i)
    {
      error_strs.insert(errors.child(i).as_string());
    }
    gather_strings(error_strs);
    conduit::Node n_errors;
    for(auto e : error_strs)
    {
      n_errors.append() = e;
    }
    ASCENT_ERROR("Jit errors: "<<n_errors.to_string());
  }
}
//-----------------------------------------------------------------------------
class ExprJitFilterFactoryFunctor
{
public:
  //---------------------------------------------------------------------------
  static void
  set(const int num_inputs,
      const std::shared_ptr<const JitExecutionPolicy> exec_policy)
  {
    // set static members
    m_num_inputs  = num_inputs;
    m_exec_policy = exec_policy;
  }

  //---------------------------------------------------------------------------
  static Filter *
  JitFilterFactoryFunctor(const std::string &filter_type_name)
  {
    // gen generic jit filter with # inputs and exec polic
    return new ExprJitFilter(m_num_inputs, m_exec_policy);
  }

private:
  static int                                       m_num_inputs;
  static std::shared_ptr<const JitExecutionPolicy> m_exec_policy;
};

//-----------------------------------------------------------------------------
// declare ExprJitFilterFactoryFunctor static members
int ExprJitFilterFactoryFunctor::m_num_inputs;
std::shared_ptr<const JitExecutionPolicy> ExprJitFilterFactoryFunctor::m_exec_policy;


//-----------------------------------------------------------------------------
std::string
register_jit_filter(flow::Workspace &w,
                    const int num_inputs,
                    const std::shared_ptr<const JitExecutionPolicy> exec_policy)
{
  ExprJitFilterFactoryFunctor::set(num_inputs, exec_policy);
  std::stringstream ss;
  ss << "expr_jit_filter_" << num_inputs << "_" << exec_policy->get_name();
  if(!w.supports_filter_type(ss.str()))
  {
    flow::Workspace::register_filter_type(
        ss.str(), ExprJitFilterFactoryFunctor::JitFilterFactoryFunctor);
  }
  return ss.str();
}

//******************************************************************************
// ExprExpressionList
//******************************************************************************

//-----------------------------------------------------------------------------
ExprExpressionList::ExprExpressionList(int num_inputs)
  : Filter(),
    m_num_inputs(num_inputs)
{
  // empty
}

//-----------------------------------------------------------------------------
ExprExpressionList::ExprExpressionList()
  : Filter(),
    m_num_inputs(256)
{
  // empty
}

//-----------------------------------------------------------------------------
ExprExpressionList::~ExprExpressionList()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ExprExpressionList::declare_interface(Node &i)
{
  i["type_name"] = "expr_expr_list";
  // We can't have an arbitrary number of input ports so we choose 256
  for(int item_num = 0; item_num < m_num_inputs; ++item_num)
  {
    std::stringstream ss;
    ss << "item_" << item_num;
    i["port_names"].append() = ss.str();
  }
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ExprExpressionList::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ExprExpressionList::execute()
{
  conduit::Node *output = new conduit::Node();

  Node &value = (*output)["value"];
  for(int item_num = 0; item_num < m_num_inputs; ++item_num)
  {
    std::stringstream ss;
    ss << "item_" << item_num;
    const conduit::Node *n_item = input<Node>(ss.str());
    if(n_item->dtype().is_empty())
    {
      break;
    }
    value.append() = *n_item;
  }
  (*output)["type"] = "list";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Filter *
ExprExpressionListFilterFactoryMethod(const std::string &filter_type_name)
{
  size_t prefix_size = std::char_traits<char>::length("expr_expr_list_");
  const std::string num_inputs_str = filter_type_name.substr(prefix_size,
                                         filter_type_name.size() - prefix_size);
  const int num_inputs = std::stoi(num_inputs_str);
  return new ExprExpressionList(num_inputs);
}

//-----------------------------------------------------------------------------
std::string
register_expression_list_filter(flow::Workspace &w, const int num_inputs)
{
  std::stringstream ss;
  ss << "expr_expr_list_" << num_inputs;
  if(!w.supports_filter_type(ss.str()))
  {
    flow::Workspace::register_filter_type(ss.str(),
                                         ExprExpressionListFilterFactoryMethod);
  }
  return ss.str();
}

};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
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
