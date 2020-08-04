#include "ascent_expressions_ast.hpp"
#include "ascent_derived_jit.hpp"
#include "ascent_expression_filters.hpp"
#include "ascent_expressions_parser.hpp"
#include <array>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

using namespace std;
/* -- Code Generation -- */

namespace detail
{
std::string
strip_single_quotes(const std::string str)
{
  std::string stripped = str;
  int pos = stripped.find("'");
  while(pos != std::string::npos)
  {
    stripped.erase(pos, 1);
    pos = stripped.find("'");
  }
  return stripped;
}

std::string
print_match_error(const std::string &fname,
                  const std::vector<conduit::Node> &pos_arg_nodes,
                  const std::vector<conduit::Node> &named_arg_nodes,
                  const std::vector<std::string> &named_arg_names,
                  const conduit::Node &overload_list)
{
  std::stringstream ss;
  ss << "Could not match function : ";
  ss << fname << "(";
  for(int i = 0; i < pos_arg_nodes.size(); ++i)
  {
    if(i > 0)
    {
      ss << ", ";
    }
    ss << pos_arg_nodes[i]["type"].as_string();
  }
  for(int i = 0; i < named_arg_nodes.size(); ++i)
  {
    ss << ", " << named_arg_names[i] << "="
       << named_arg_nodes[i]["type"].as_string();
  }
  ss << ")" << std::endl;

  ss << "Known function signatures :\n";
  for(int i = 0; i < overload_list.number_of_children(); ++i)
  {
    ss << " " << fname << "(";
    const conduit::Node &sig = overload_list.child(i);
    const conduit::Node &n_args = sig["args"];
    const int req_args = sig["req_count"].to_int32();
    for(int a = 0; a < n_args.number_of_children(); ++a)
    {
      if(a >= req_args)
      {
        ss << "[optional]";
      }
      ss << n_args.child(a).name() << "="
         << n_args.child(a)["type"].as_string();
      if(a == n_args.number_of_children() - 1)
      {
        ss << ")\n";
      }
      else
      {
        ss << ", ";
      }
    }
  }

  return ss.str();
}

// TODO these functions are duplicated from ascent_expression_filters.cpp
bool
is_math(const std::string &op)
{
  return op == "*" || op == "+" || op == "/" || op == "-" || op == "%";
}

bool
is_logic(const std::string &op)
{
  return op == "or" || op == "and" || op == "not";
}

bool
is_scalar(const std::string &type)
{
  return type == "int" || type == "double" || type == "scalar";
}

bool
is_field_type(const std::string &type)
{
  return type == "field" || type == "jitable";
}
} // namespace detail

//-----------------------------------------------------------------------------
void
ASTExpression::access()
{
  std::cout << "placeholder expression" << std::endl;
}

conduit::Node
ASTExpression::build_graph(flow::Workspace &w, bool verbose)
{
  // placeholder to make binary op work with "not"
  if(!w.graph().has_filter("bop_placeholder"))
  {
    conduit::Node placeholder_params;
    placeholder_params["value"] = true;
    w.graph().add_filter("expr_bool", "bop_placeholder", placeholder_params);
  }
  conduit::Node res;
  res["filter_name"] = "bop_placeholder";
  res["type"] = "bool";
  return res;
}

//-----------------------------------------------------------------------------
void
ASTInteger::access()
{
  std::cout << "Creating integer: " << m_value << endl;
}

conduit::Node
ASTInteger::build_graph(flow::Workspace &w, bool verbose)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  // create a unique name for each expression so we can reuse subexpressions
  std::stringstream ss;
  ss << "integer_" << m_value;
  const std::string verbose_name = ss.str();
  if((*subexpr_cache).has_path(verbose_name))
  {
    return (*subexpr_cache)[verbose_name];
  }

  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    static int integer_counter = 0;
    std::stringstream ss;
    ss << "integer_" << integer_counter++;
    name = ss.str();
  }

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_integer", name, params);
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "int";
  (*subexpr_cache)[verbose_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTDouble::access()
{
  std::cout << "Creating double: " << m_value << endl;
}

conduit::Node
ASTDouble::build_graph(flow::Workspace &w, bool verbose)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "double_" << m_value;
  const std::string verbose_name = ss.str();
  if((*subexpr_cache).has_path(verbose_name))
  {
    return (*subexpr_cache)[verbose_name];
  }

  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    static int double_counter = 0;
    std::stringstream ss;
    ss << "double_" << double_counter++;
    name = ss.str();
  }

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_double", name, params);

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "double";
  (*subexpr_cache)[verbose_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTIdentifier::access()
{
  std::cout << "Creating identifier reference: " << m_name << endl;
}

conduit::Node
ASTIdentifier::build_graph(flow::Workspace &w, bool verbose)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "ident_" << m_name;
  const std::string name = ss.str();
  if((*subexpr_cache).has_path(name))
  {
    return (*subexpr_cache)[name];
  }

  // if (context.locals().find(name) == context.locals().end())
  //{
  //  std::cerr << "undeclared variable " << name << endl;
  //  return NULL;
  //}
  // return new LoadInst(context.locals()[name], "", false,
  // context.currentBlock());

  conduit::Node params;
  params["value"] = m_name;

  w.graph().add_filter("expr_identifier", name, params);

  conduit::Node res;
  res["filter_name"] = name;

  // get identifier type from cache
  conduit::Node *cache = w.registry().fetch<conduit::Node>("cache");
  if(!cache->has_path(m_name))
  {
    ASCENT_ERROR("Unknown expression identifier: '" << m_name << "'");
  }

  const int entries = (*cache)[m_name].number_of_children();
  if(entries < 1)
  {
    ASCENT_ERROR("Expression identifier: needs a non-zero number of entires: "
                 << entries);
  }
  // grab the last one calculated
  res["type"] = (*cache)[m_name].child(entries - 1)["type"];
  (*subexpr_cache)[name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTNamedExpression::access()
{
  key->access();
  value->access();
}

conduit::Node
ASTNamedExpression::build_graph(flow::Workspace &w, bool verbose)
{
  return value->build_graph(w, verbose);
}

//-----------------------------------------------------------------------------
void
ASTArguments::access()
{
  if(pos_args != nullptr)
  {
    std::cout << "Creating positional arguments" << std::endl;
    const size_t pos_size = pos_args->exprs.size();
    for(size_t i = 0; i < pos_size; ++i)
    {
      pos_args->exprs[i]->access();
    }
  }

  if(named_args != nullptr)
  {
    std::cout << "Creating named arguments" << std::endl;
    const size_t named_size = named_args->size();
    for(size_t i = 0; i < named_size; ++i)
    {
      (*named_args)[i]->access();
    }
  }
}

//-----------------------------------------------------------------------------
void
ASTMethodCall::access()
{
  std::cout << "Creating method call: " << m_id->m_name << std::endl;
  arguments->access();
}

conduit::Node
ASTMethodCall::build_graph(flow::Workspace &w, bool verbose)
{
  // build the positional arguments
  size_t pos_size = 0;
  std::vector<conduit::Node> pos_arg_nodes;
  if(arguments->pos_args != nullptr)
  {
    pos_size = arguments->pos_args->exprs.size();
    pos_arg_nodes.resize(pos_size);
    for(size_t i = 0; i < pos_size; ++i)
    {
      pos_arg_nodes[i] = arguments->pos_args->exprs[i]->build_graph(w, verbose);
    }
  }

  // build the named arguments
  size_t named_size = 0;
  std::vector<conduit::Node> named_arg_nodes;
  // stores argument names in the same order as named_arg_nodes
  std::vector<std::string> named_arg_names;
  if(arguments->named_args != nullptr)
  {
    named_size = arguments->named_args->size();
    named_arg_nodes.resize(named_size);
    named_arg_names.resize(named_size);
    for(size_t i = 0; i < named_size; ++i)
    {
      named_arg_nodes[i] = (*arguments->named_args)[i]->build_graph(w, verbose);
      named_arg_names[i] = (*arguments->named_args)[i]->key->m_name;
    }
  }

  if(!w.registry().has_entry("function_table"))
  {
    ASCENT_ERROR("Missing function table");
  }

  conduit::Node *f_table = w.registry().fetch<conduit::Node>("function_table");
  // resolve the function
  if(!f_table->has_path(m_id->m_name))
  {
    ASCENT_ERROR("Expressions: Unknown function '" << m_id->m_name << "'.");
  }

  // resolve overloaded function names
  const conduit::Node &overload_list = (*f_table)[m_id->m_name];

  int matched_index = -1;

  std::unordered_set<std::string> req_args;
  std::unordered_set<std::string> opt_args;
  // keeps track of how things are connected
  std::map<std::string, const conduit::Node *> args_map;
  for(int i = 0; i < overload_list.number_of_children(); ++i)
  {
    const conduit::Node &func = overload_list.child(i);
    bool valid = true;
    int total_args = 0;

    if(func.has_path("args"))
    {
      total_args = func["args"].number_of_children();
    }

    // validation
    if(named_size + pos_size > total_args)
    {
      continue;
    }

    const std::vector<std::string> func_arg_names = func["args"].child_names();

    // populate args_map
    args_map.clear();
    for(int a = 0; a < pos_size; ++a)
    {
      const conduit::Node &arg = pos_arg_nodes[a];
      args_map[func_arg_names[a]] = &arg;
    }
    for(int a = 0; a < named_size; ++a)
    {
      const conduit::Node &arg = named_arg_nodes[a];
      // ensure an argument wasn't passed twice
      if(args_map.find(named_arg_names[a]) == args_map.end())
      {
        args_map[named_arg_names[a]] = &arg;
      }
      else
      {
        valid = false;
      }
    }
    if(!valid)
    {
      continue;
    }

    // populate opt_args and req_args
    req_args.clear();
    opt_args.clear();
    for(int a = 0; a < total_args; ++a)
    {
      conduit::Node func_arg = func["args"].child(a);
      if(func_arg.has_path("optional"))
      {
        opt_args.insert(func_arg_names[a]);
      }
      else
      {
        req_args.insert(func_arg_names[a]);
      }
    }

    // validate arg types
    for(auto const &arg : args_map)
    {
      // check if argument name exists
      if(!func["args"].has_path(arg.first))
      {
        valid = false;
        break;
      }
      const conduit::Node &sig_arg = func["args/" + arg.first];
      const std::string expected_type = sig_arg["type"].as_string();
      const std::string passed_type = (*arg.second)["type"].as_string();

      // validate types
      if(!(detail::is_field_type(expected_type) &&
           detail::is_field_type(passed_type)) &&
         !(expected_type == "scalar" && detail::is_scalar(passed_type)) &&
         expected_type != "anytype" && passed_type != expected_type)
      {
        valid = false;
        break;
      }

      // keep track of which arguments have been specified
      if(sig_arg.has_path("optional"))
      {
        valid &= opt_args.erase(arg.first);
      }
      else
      {
        valid &= req_args.erase(arg.first);
      }
      if(!valid)
      {
        break;
      }
    }
    if(!valid)
    {
      continue;
    }

    // ensure all required arguments are passed
    if(!req_args.empty())
    {
      continue;
    }

    // we made it to the end!
    matched_index = i;
    break;
  }

  conduit::Node res;

  if(matched_index != -1)
  {
    const conduit::Node &func = overload_list.child(matched_index);

    conduit::Node *subexpr_cache =
        w.registry().fetch<conduit::Node>("subexpr_cache");
    std::stringstream ss;
    std::string name;
    std::string verbose_name;
    if(func.has_path("jitable"))
    {
      // jit case
      ss << "jit_method_" << func["filter_name"].as_string() << "(";
      bool first = true;
      for(auto const &arg : args_map)
      {
        if(!first)
        {
          ss << ", ";
        }
        ss << arg.first << "=" << (*arg.second)["filter_name"].as_string();
        first = false;
      }
      ss << ")";
      verbose_name = ss.str();
      if((*subexpr_cache).has_path(verbose_name))
      {
        return (*subexpr_cache)[verbose_name];
      }
      if(verbose)
      {
        name = verbose_name;
      }
      else
      {
        std::stringstream ss;
        static int jit_method_counter = 0;
        ss << "jit_method_" << jit_method_counter++ << "_"
           << func["filter_name"].as_string();

        name = ss.str();
      }

      // generate params
      conduit::Node params;
      params["func"] = func["filter_name"].as_string();
      params["filter_name"] = name;
      params["execute"] = false;
      int port = 0;
      for(auto const &arg : args_map)
      {
        conduit::Node &inp = params["inputs/" + arg.first];
        inp = *arg.second;
        inp["port"] = port;
        ++port;
      }

      w.graph().add_filter(
          ascent::runtime::expressions::register_jit_filter(w, args_map.size()),
          name,
          params);

      // connect up all the arguments
      port = 0;
      for(auto const &arg : args_map)
      {
        // src, dest, port
        w.graph().connect((*arg.second)["filter_name"].as_string(), name, port);
        ++port;
      }
    }
    else
    {
      // non-jit case
      ss << "method_" << func["filter_name"].as_string() << "(";
      bool first = true;
      for(auto const &arg : args_map)
      {
        if(!first)
        {
          ss << ", ";
        }
        ss << arg.first << "=" << (*arg.second)["filter_name"].as_string();
        first = false;
      }
      ss << ")";
      verbose_name = ss.str();
      if((*subexpr_cache).has_path(verbose_name))
      {
        return (*subexpr_cache)[verbose_name];
      }
      if(verbose)
      {
        name = verbose_name;
      }
      else
      {
        std::stringstream ss;
        static int method_counter = 0;
        ss << "method_" << method_counter++ << "_"
           << func["filter_name"].as_string();
        name = ss.str();
      }

      // we will have some optional parameters, prep the null_args filter
      if(!opt_args.empty())
      {
        if(!w.graph().has_filter("null_arg"))
        {
          conduit::Node null_params;
          w.graph().add_filter("null_arg", "null_arg", null_params);
        }
      }

      conduit::Node params;
      w.graph().add_filter(func["filter_name"].as_string(), name, params);

      // connect up all the arguments
      for(auto const &arg : args_map)
      {
        std::string inp_filter_name = (*arg.second)["filter_name"].as_string();
        // we must to execute inputs that are jitables if the function is
        // not jitable
        if((*arg.second)["type"].as_string() == "jitable")
        {
          // create a unique name for the filter
          std::stringstream ss;
          ss << "jit_method_execute_" << inp_filter_name;
          const std::string jit_execute_name = ss.str();
          if(!(*subexpr_cache).has_path(jit_execute_name))
          {
            conduit::Node params;
            params["func"] = "execute";
            params["filter_name"] = jit_execute_name;
            params["execute"] = true;
            conduit::Node &inp = params["inputs/jitable"];
            inp = *arg.second;
            inp["port"] = 0;
            w.graph().add_filter(
                ascent::runtime::expressions::register_jit_filter(w, 1),
                jit_execute_name,
                params);
            // src, dest, port
            w.graph().connect(inp_filter_name, jit_execute_name, 0);
          }
          inp_filter_name = jit_execute_name;
        }
        // src, dest, port
        w.graph().connect(inp_filter_name, name, arg.first);
      }

      // connect null filter to optional args that weren't passed in
      for(std::unordered_set<std::string>::iterator it = opt_args.begin();
          it != opt_args.end();
          ++it)
      {
        w.graph().connect("null_arg", name, *it);
      }
    }

    // evaluate what the return type will be
    std::string res_type = func["return_type"].as_string();
    // we will need special code (per function) to determine the correct return
    // type
    if(res_type == "anytype")
    {
      // the history function's return type is the type of its first argument
      if(func["filter_name"].as_string() == "history")
      {
        res_type = (*args_map["expr_name"])["type"].as_string();
      }
      else
      {
        ASCENT_ERROR("Could not determine the return type of "
                     << func["filter_name"].as_string());
      }
    }

    res["filter_name"] = name;
    res["type"] = res_type;
    (*subexpr_cache)[verbose_name] = res;
  }
  else
  {
    ASCENT_ERROR(detail::print_match_error(m_id->m_name,
                                           pos_arg_nodes,
                                           named_arg_nodes,
                                           named_arg_names,
                                           overload_list));
  }

  return res;
}

//-----------------------------------------------------------------------------
void
ASTIfExpr::access()
{
  m_condition->access();
  std::cout << "Creating if condition" << std::endl;

  m_if->access();
  std::cout << "Creating if body" << std::endl;

  m_else->access();
  std::cout << "Creating else body" << std::endl;

  std::cout << "Creating if expression" << std::endl;
}

conduit::Node
ASTIfExpr::build_graph(flow::Workspace &w, bool verbose)
{
  conduit::Node n_condition = m_condition->build_graph(w, verbose);
  conduit::Node n_if = m_if->build_graph(w, verbose);
  conduit::Node n_else = m_else->build_graph(w, verbose);

  // Validate types
  const std::string condition_type = n_condition["type"].as_string();
  const std::string if_type = n_if["type"].as_string();
  const std::string else_type = n_else["type"].as_string();
  std::string res_type;
  if(detail::is_field_type(condition_type))
  {
    if((!detail::is_scalar(if_type) && !detail::is_field_type(if_type)) ||
       (!detail::is_scalar(else_type) && !detail::is_field_type(else_type)))
    {
      ASCENT_ERROR(
          "If the if-condition is a field type then the if and else branches "
          "must return scalars or field types. condition_type: '"
          << condition_type << "', if_type: '" << if_type << "', else_type: '"
          << else_type << "'.");
    }
    res_type = "jitable";
  }
  else if(condition_type == "bool")
  {
    if(if_type != else_type)
    {
      ASCENT_ERROR("The return types of the if (" << if_type << ") and else ("
                                                  << else_type
                                                  << ") branches must match");
    }
    res_type = if_type;
  }
  else
  {
    ASCENT_ERROR("if-expression condition is of type: '"
                 << condition_type
                 << "' but must be of type 'bool' or a field type.");
  }

  std::string name;
  std::string verbose_name;
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  if(res_type == "jitable")
  {
    std::stringstream ss;
    ss << "jit_if_" << n_condition["filter_name"].as_string() << "_then_"
       << n_if["filter_name"].as_string() << "_else_"
       << n_else["filter_name"].as_string();
    verbose_name = ss.str();
    if((*subexpr_cache).has_path(verbose_name))
    {
      return (*subexpr_cache)[verbose_name];
    }
    if(verbose)
    {
      name = verbose_name;
    }
    else
    {
      static int jit_if_counter = 0;
      std::stringstream ss;
      ss << "jit_if_" << jit_if_counter++;
      name = ss.str();
    }
    conduit::Node params;
    params["func"] = "expr_if";
    params["filter_name"] = name;
    params["execute"] = false;
    conduit::Node &condition = params["inputs/condition"];
    condition = n_condition;
    condition["port"] = 0;
    conduit::Node &p_if = params["inputs/if"];
    p_if = n_if;
    p_if["port"] = 1;
    conduit::Node &p_else = params["inputs/else"];
    p_else = n_else;
    p_else["port"] = 2;

    w.graph().add_filter(
        ascent::runtime::expressions::register_jit_filter(w, 3), name, params);
    // src, dest, port
    w.graph().connect(n_condition["filter_name"].as_string(), name, 0);
    w.graph().connect(n_if["filter_name"].as_string(), name, 1);
    w.graph().connect(n_else["filter_name"].as_string(), name, 2);
  }
  else
  {
    std::stringstream ss;
    ss << "if_" << n_condition["filter_name"].as_string() << "_then_"
       << n_if["filter_name"].as_string() << "_else_"
       << n_else["filter_name"].as_string();
    verbose_name = ss.str();
    if((*subexpr_cache).has_path(verbose_name))
    {
      return (*subexpr_cache)[verbose_name];
    }
    if(verbose)
    {
      name = verbose_name;
    }
    else
    {
      static int if_counter = 0;
      std::stringstream ss;
      ss << "if_" << if_counter++;
      name = ss.str();
    }

    conduit::Node params;
    w.graph().add_filter("expr_if", name, params);

    // src, dest, port
    w.graph().connect(
        n_condition["filter_name"].as_string(), name, "condition");
    w.graph().connect(n_if["filter_name"].as_string(), name, "if");
    w.graph().connect(n_else["filter_name"].as_string(), name, "else");
  }
  conduit::Node res;
  res["type"] = res_type;
  res["filter_name"] = name;
  (*subexpr_cache)[verbose_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTBinaryOp::access()
{
  // std::cout << "Creating binary operation " << m_op << endl;
  // Instruction::BinaryOps instr;
  std::string op_str;
  switch(m_op)
  {
  case TPLUS: op_str = "+"; break;
  case TMINUS: op_str = "-"; break;
  case TMUL: op_str = "*"; break;
  case TDIV: op_str = "/"; break;
  case TMOD: op_str = "%"; break;
  case TCEQ: op_str = "=="; break;
  case TCNE: op_str = "!="; break;
  case TCLE: op_str = "<="; break;
  case TCGE: op_str = ">="; break;
  case TCGT: op_str = ">"; break;
  case TCLT: op_str = "<"; break;
  case TOR: op_str = "or"; break;
  case TAND: op_str = "and"; break;
  case TNOT: op_str = "not"; break;
  default: ASCENT_ERROR("unknown binary op " << m_op);
  }

  m_rhs->access();
  // std::cout << " op " << op_str << "\n";
  m_lhs->access();
}

conduit::Node
ASTBinaryOp::build_graph(flow::Workspace &w, const bool verbose)
{
  // std::cout << "Creating binary operation " << m_op << endl;
  std::string op_str;
  switch(m_op)
  {
  case TPLUS: op_str = "+"; break;
  case TMINUS: op_str = "-"; break;
  case TMUL: op_str = "*"; break;
  case TDIV: op_str = "/"; break;
  case TMOD: op_str = "%"; break;
  case TCEQ: op_str = "=="; break;
  case TCNE: op_str = "!="; break;
  case TCLE: op_str = "<="; break;
  case TCGE: op_str = ">="; break;
  case TCGT: op_str = ">"; break;
  case TCLT: op_str = "<"; break;
  case TOR: op_str = "or"; break;
  case TAND: op_str = "and"; break;
  case TNOT: op_str = "not"; break;
  default: ASCENT_ERROR("unknown binary op " << m_op);
  }

  conduit::Node l_in = m_lhs->build_graph(w, verbose);
  // std::cout << " flow op " << op_str << "\n";
  conduit::Node r_in = m_rhs->build_graph(w, verbose);

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  // flow doesn't like it when we have a / in the filter name
  ss << "binary_op"
     << "(" << l_in["filter_name"].as_string()
     << (op_str == "/" ? "div" : op_str) << r_in["filter_name"].as_string()
     << ")";
  const std::string verbose_name = ss.str();
  if((*subexpr_cache).has_path(verbose_name))
  {
    return (*subexpr_cache)[verbose_name];
  }
  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    std::stringstream ss;
    static int binary_op_counter = 0;
    ss << "binary_op_" << binary_op_counter++;
    name = ss.str();
  }

  // Validate types and evaluate what the return type will be
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();

  std::string res_type;
  if(detail::is_math(op_str))
  {
    if((detail::is_scalar(l_type) && detail::is_field_type(r_type)) ||
       (detail::is_scalar(r_type) && detail::is_field_type(l_type)) ||
       (detail::is_field_type(l_type) && detail::is_field_type(r_type)))
    {
      res_type = "jitable";
    }
    else if(detail::is_scalar(l_type) && detail::is_scalar(r_type))
    {
      // promote to double if at least one is a double
      if(l_type == "double" || r_type == "double")
      {
        res_type = "double";
      }
      else
      {
        res_type = "int";
      }
    }
    else if(l_type == "vector" && r_type == "vector")
    {
      res_type = "vector";
    }
    else
    {
      ASCENT_ERROR("Unsupported math operation: "
                   << "'" << l_type << " " << op_str << " " << r_type << "'");
    }
  }
  else if(detail::is_logic(op_str))
  {
    if((l_type == "bool" && detail::is_field_type(r_type)) ||
       (r_type == "bool" && detail::is_field_type(l_type)) ||
       (detail::is_field_type(l_type) && detail::is_field_type(r_type)))
    {
      res_type = "jitable";
    }
    else if(l_type != "bool" || r_type != "bool")
    {
      ASCENT_ERROR(
          "logical operators are only supported on bools and field types: "
          << "'" << l_type << " " << op_str << " " << r_type << "'");
    }
    else
    {
      res_type = "bool";
    }
  }
  else
  {
    // comparison ops
    if((detail::is_scalar(l_type) && detail::is_field_type(r_type)) ||
       (detail::is_scalar(r_type) && detail::is_field_type(l_type)) ||
       (detail::is_field_type(l_type) && detail::is_field_type(r_type)))
    {
      res_type = "jitable";
    }
    else if(!detail::is_scalar(l_type) || !detail::is_scalar(r_type))
    {
      ASCENT_ERROR(
          "comparison operators are only supported on scalars and field types: "
          << "'" << l_type << " " << op_str << " " << r_type << "'");
    }
    else
    {
      res_type = "bool";
    }
  }

  if(res_type == "jitable")
  {

    conduit::Node params;
    params["func"] = "binary_op";
    params["filter_name"] = name;
    params["execute"] = false;
    params["op_string"] = op_str;
    conduit::Node &l_param = params["inputs/lhs"];
    l_param = l_in;
    l_param["port"] = 0;
    conduit::Node &r_param = params["inputs/rhs"];
    r_param = r_in;
    r_param["port"] = 1;

    w.graph().add_filter(
        ascent::runtime::expressions::register_jit_filter(w, 2), name, params);

    // src, dest, port
    w.graph().connect(l_in["filter_name"].as_string(), name, 0);
    w.graph().connect(r_in["filter_name"].as_string(), name, 1);
  }
  else
  {
    conduit::Node params;
    params["op_string"] = op_str;

    w.graph().add_filter("expr_binary_op", name, params);

    // src, dest, port
    w.graph().connect(l_in["filter_name"].as_string(), name, "lhs");
    w.graph().connect(r_in["filter_name"].as_string(), name, "rhs");
  }
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = res_type;
  (*subexpr_cache)[verbose_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTString::access()
{
  std::cout << "Creating string " << m_name << endl;
}

conduit::Node
ASTString::build_graph(flow::Workspace &w, bool verbose)
{
  // strip the quotes from the variable name
  std::string stripped = detail::strip_single_quotes(m_name);

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "string_" << stripped;
  const std::string name = ss.str();
  if((*subexpr_cache).has_path(name))
  {
    return (*subexpr_cache)[name];
  }

  conduit::Node params;
  params["value"] = stripped;

  w.graph().add_filter("expr_string", name, params);

  conduit::Node res;
  res["type"] = "string";
  res["filter_name"] = name;
  (*subexpr_cache)[name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTBoolean::access()
{
  std::string bool_str = "";
  switch(tok)
  {
  case TTRUE: bool_str = "True"; break;
  case TFALSE: bool_str = "False"; break;
  default: std::cout << "unknown bool literal " << tok << "\n";
  }
  std::cout << "Creating bool literal " << bool_str << std::endl;
}

conduit::Node
ASTBoolean::build_graph(flow::Workspace &w, bool verbose)
{
  bool value = false;
  switch(tok)
  {
  case TTRUE: value = true; break;
  case TFALSE: value = false; break;
  default: std::cout << "unknown bool literal " << tok << "\n";
  }

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "bool_" << value;
  const std::string name = ss.str();
  if((*subexpr_cache).has_path(name))
  {
    return (*subexpr_cache)[name];
  }

  conduit::Node params;
  params["value"] = value;
  w.graph().add_filter("expr_bool", name, params);

  conduit::Node res;
  res["type"] = "bool";
  res["filter_name"] = name;
  (*subexpr_cache)[name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTArrayAccess::access()
{
  array->access();
  std::cout << "Creating array" << std::endl;

  index->access();
  std::cout << "Creating array index" << std::endl;

  std::cout << "Creating array access" << std::endl;
}

conduit::Node
ASTArrayAccess::build_graph(flow::Workspace &w, bool verbose)
{
  conduit::Node n_array = array->build_graph(w, verbose);
  conduit::Node n_index = index->build_graph(w, verbose);

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "array_access_" << n_array["filter_name"].as_string() << "["
     << n_index["filter_name"].as_string() << "]";
  const std::string name = ss.str();
  if((*subexpr_cache).has_path(name))
  {
    return (*subexpr_cache)[name];
  }

  conduit::Node params;

  if(n_index["type"].as_string() != "int")
  {
    ASCENT_ERROR("Array index must be an integer");
  }

  std::string obj_type = n_array["type"].as_string();
  if(obj_type != "array")
  {
    ASCENT_ERROR("Cannot get index of non-array type: " << obj_type);
  }

  w.graph().add_filter("expr_array", name, params);

  // src, dest, port
  w.graph().connect(n_array["filter_name"].as_string(), name, "array");
  w.graph().connect(n_index["filter_name"].as_string(), name, "index");

  conduit::Node res;
  // only arrays of double are supported
  res["type"] = "double";
  res["filter_name"] = name;
  (*subexpr_cache)[name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTDotAccess::access()
{
  obj->access();
  std::cout << "Creating object" << std::endl;

  std::cout << "Creating dot name " << name << std::endl;

  std::cout << "Creating dot access" << std::endl;
}

conduit::Node
ASTDotAccess::build_graph(flow::Workspace &w, bool verbose)
{
  const conduit::Node input_obj = obj->build_graph(w, verbose);

  std::string obj_type = input_obj["type"].as_string();

  // load the object table
  if(!w.registry().has_entry("object_table"))
  {
    ASCENT_ERROR("Missing object table");
  }
  conduit::Node *o_table = w.registry().fetch<conduit::Node>("object_table");

  // get the object
  if(!o_table->has_path(obj_type))
  {
    ASCENT_ERROR("Cannot get attribute of non-object type: " << obj_type);
  }
  const conduit::Node &o_table_obj = (*o_table)[obj_type];

  // resolve attribute type
  std::string path = "attrs/" + name + "/type";
  if(!o_table_obj.has_path(path))
  {
    std::stringstream ss;
    if(o_table_obj.has_path("attrs"))
    {
      std::string attr_yaml = o_table_obj["attrs"].to_yaml();
      if(attr_yaml == "")
      {
        ss << " No know attribtues.";
      }
      else
      {
        ss << " Known attributes: " << attr_yaml;
      }
    }
    else
    {
      ss << " No known attributes.";
    }
    ASCENT_ERROR("Attribute " << name << " of " << obj_type << " not found."
                              << ss.str());
  }
  std::string res_type = o_table_obj[path].as_string();

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::string f_name;
  std::string verbose_name;
  if(o_table->has_path(name + "/jitable") || res_type == "jitable")
  {
    std::stringstream ss;
    ss << "jit_dot_(" << input_obj["filter_name"].as_string() << ")__" << name;
    verbose_name = ss.str();
    if((*subexpr_cache).has_path(verbose_name))
    {
      return (*subexpr_cache)[verbose_name];
    }
    if(verbose)
    {
      f_name = verbose_name;
    }
    else
    {
      static int jit_dot_counter = 0;
      std::stringstream ss;
      ss << "jit_dot_" << jit_dot_counter++ << "__" << name;
      f_name = ss.str();
    }

    conduit::Node jit_filter_obj = input_obj;
    if(o_table_obj.has_path("jitable"))
    {
      jit_filter_obj["type"] = "jitable";
    }
    conduit::Node params;
    params["func"] = "expr_dot";
    params["filter_name"] = f_name;
    params["execute"] = false;
    params["inputs/obj"] = jit_filter_obj;
    params["inputs/obj/port"] = 0;
    params["name"] = name;
    w.graph().add_filter(
        ascent::runtime::expressions::register_jit_filter(w, 1),
        f_name,
        params);
    // src, dest, port
    w.graph().connect(input_obj["filter_name"].as_string(), f_name, 0);
  }
  else
  {
    std::stringstream ss;
    ss << "dot_(" << input_obj["filter_name"].as_string() << ")__" << name;
    verbose_name = ss.str();
    if((*subexpr_cache).has_path(verbose_name))
    {
      return (*subexpr_cache)[verbose_name];
    }
    if(verbose)
    {
      f_name = verbose_name;
    }
    else
    {
      static int dot_counter = 0;
      std::stringstream ss;
      ss << "dot_" << dot_counter++ << "__" << name;
      f_name = ss.str();
    }

    conduit::Node params;
    params["name"] = name;

    w.graph().add_filter("expr_dot", f_name, params);
    // src, dest, port
    w.graph().connect(input_obj["filter_name"].as_string(), f_name, "obj");
  }

  conduit::Node res;
  res["type"] = res_type;
  res["filter_name"] = f_name;
  (*subexpr_cache)[verbose_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTExpressionList::access()
{
  std::cout << "Creating list" << std::endl;
  for(auto expr : exprs)
  {
    expr->access();
  }
}

conduit::Node
ASTExpressionList::build_graph(flow::Workspace &w, bool verbose)
{
  const size_t list_size = exprs.size();

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ss;
  ss << "list_[";
  std::vector<conduit::Node> items;
  for(size_t i = 0; i < list_size; ++i)
  {
    items.push_back(exprs[i]->build_graph(w, verbose));
    ss << items.back()["filter_name"].as_string();
    if(i < list_size - 1)
    {
      ss << ", ";
    }
  }
  ss << "]";
  const std::string verbose_name = ss.str();
  if((*subexpr_cache).has_path(verbose_name))
  {
    return (*subexpr_cache)[verbose_name];
  }
  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    static int list_counter = 0;
    std::stringstream ss;
    ss << "list_" << list_counter++;
    name = ss.str();
  }

  conduit::Node params;
  w.graph().add_filter(
      ascent::runtime::expressions::register_expression_list_filter(w,
                                                                    list_size),
      name,
      params);

  // Connect all items to the list
  for(size_t i = 0; i < list_size; ++i)
  {
    // src, dest, port
    w.graph().connect(items[i]["filter_name"].as_string(), name, i);
  }

  conduit::Node res;
  res["type"] = "list";
  res["filter_name"] = name;
  (*subexpr_cache)[verbose_name] = res;
  return res;
}
