#include "ascent_expressions_ast.hpp"
#include "ascent_derived_jit.hpp"
#include "ascent_expressions_parser.hpp"
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
    ss << ((i == 0) ? "" : ", ") << pos_arg_nodes[i]["type"].as_string();
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
} // namespace detail

//-----------------------------------------------------------------------------
void
ASTExpression::access()
{
  std::cout << "placeholder expression" << std::endl;
}

conduit::Node
ASTExpression::build_graph(flow::Workspace &w)
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

std::string
ASTInteger::build_jit(conduit::Node &n, flow::Workspace &w)
{
  std::stringstream ss;
  // force everthing to a double
  ss << "(double)" << m_value;
  return ss.str();
}

conduit::Node
ASTInteger::build_jit2(flow::Workspace &w)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  // create a unique name for each expression so we can reuse subexpressions
  std::stringstream ess;
  ess << "jit_integer"
      << "_" << m_value;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int name_counter = 0;
  std::stringstream ss;
  ss << "jit_integer"
     << "_" << name_counter++;
  const std::string name = ss.str();

  conduit::Node params;
  std::stringstream ss2;
  // force everthing to a double
  ss2 << "(double)" << m_value;
  params["value"] = ss.str();
  w.graph().add_filter("jit_filter", name, params);
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "int";
  (*subexpr_cache)[expr_name] = res;
  return res;
}

bool
ASTInteger::can_jit()
{
  return true;
}

conduit::Node
ASTInteger::build_graph(flow::Workspace &w)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  // create a unique name for each expression so we can reuse subexpressions
  std::stringstream ess;
  ess << "integer"
      << "_" << m_value;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int ast_int_counter = 0;
  std::stringstream ss;
  ss << "integer"
     << "_" << ast_int_counter++;
  const std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_integer", name, params);
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "int";
  (*subexpr_cache)[expr_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
std::string
ASTDouble::build_jit(conduit::Node &n, flow::Workspace &w)
{
  std::stringstream ss;
  // force everthing to a double
  ss << "(double)" << m_value;
  return ss.str();
}

//-----------------------------------------------------------------------------
conduit::Node
ASTDouble::build_jit2(flow::Workspace &w)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  // create a unique name for each expression so we can reuse subexpressions
  std::stringstream ess;
  ess << "jit_double"
      << "_" << m_value;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int name_counter = 0;
  std::stringstream ss;
  ss << "jit_double"
     << "_" << name_counter++;
  const std::string name = ss.str();

  conduit::Node params;
  std::stringstream ss2;
  // force everthing to a double
  ss2 << "(double)" << m_value;
  params["value"] = ss.str();
  w.graph().add_filter("jit_filter", name, params);
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "double";
  (*subexpr_cache)[expr_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
bool
ASTDouble::can_jit()
{
  return true;
}

//-----------------------------------------------------------------------------
void
ASTDouble::access()
{
  std::cout << "Creating double: " << m_value << endl;
}

conduit::Node
ASTDouble::build_graph(flow::Workspace &w)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "double"
      << "_" << m_value;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // std::cout << "Flow double: " << m_value << endl;
  static int ast_double_counter = 0;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "double"
     << "_" << ast_double_counter++;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_double", name, params);

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "double";
  (*subexpr_cache)[expr_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTIdentifier::access()
{
  std::cout << "Creating identifier reference: " << m_name << endl;
}

std::string
ASTIdentifier::build_jit(conduit::Node &n, flow::Workspace &w)
{
  return m_name;
}

bool
ASTIdentifier::can_jit()
{
  // Maybe?
  return true;
}

conduit::Node
ASTIdentifier::build_graph(flow::Workspace &w)
{
  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "ident"
      << "_" << m_name;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // if (context.locals().find(name) == context.locals().end())
  //{
  //  std::cerr << "undeclared variable " << name << endl;
  //  return NULL;
  //}
  // return new LoadInst(context.locals()[name], "", false,
  // context.currentBlock());

  static int ast_ident_counter = 0;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "ident"
     << "_" << ast_ident_counter++;
  std::string name = ss.str();

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
  (*subexpr_cache)[expr_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTNamedExpression::access()
{
  key->access();
  value->access();
}

std::string
ASTNamedExpression::build_jit(conduit::Node &n, flow::Workspace &w)
{
  return key->build_jit(n, w) + value->build_jit(n, w);
}

conduit::Node
ASTNamedExpression::build_graph(flow::Workspace &w)
{
  return value->build_graph(w);
}

//-----------------------------------------------------------------------------
bool
ASTArguments::can_jit()
{
  bool res = true;
  if(pos_args != nullptr)
  {
    const size_t pos_size = pos_args->exprs.size();
    for(size_t i = 0; i < pos_size; ++i)
    {
      res &= pos_args->exprs[i]->can_jit();
    }
  }

  if(named_args != nullptr)
  {
    const size_t named_size = named_args->size();
    // we can't jit named
    if(named_size > 0)
    {
      res = false;
    }
  }
  return res;
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

bool
ASTMethodCall::can_jit()
{
  // TODO: validate special functions
  // and supported math funtions (max, sin, etc)

  // There are three types of functions.
  // 1) special functions: field variables that become
  //    pointers inside kernels and constants like
  //    domain_id
  // 2) pre-execute functions: max(field('braid')),
  //    and objects like histogram(field('braid),stuff
  //    can also be executed breofre hand and subbed in
  //    Note: allowing runtime expressions as paramters
  //    into 'bins' of a histogram would need to break
  //    up the kernel into different invocations.
  // 3) math functions: ceil, sin, max,...
  //    we can just pass these into jit

  // special functions
  size_t arg_size = 0;
  if(arguments->pos_args != nullptr)
  {
    arg_size = arguments->pos_args->exprs.size();
  }
  bool res = false;
  if(m_id->m_name == "field" && arg_size == 1)
  {
    res = true;
  }

  if(m_id->m_name == "domain_id" && arg_size == 0)
  {
    res = true;
  }

  // pre-execute functions
  if(m_id->m_name == "histogram")
  {
    res = false;
  }
  if(m_id->m_name == "max" && arg_size == 1)
  {
    res = true;
  }

  // math functions
  if(m_id->m_name == "max" && arg_size == 2)
  {
    res = true;
  }
  if(m_id->m_name == "min" && arg_size == 2)
  {
    res = true;
  }

  return res;
}

std::string
ASTMethodCall::build_jit(conduit::Node &n, flow::Workspace &w)
{
  size_t arg_size = 0;
  if(arguments->pos_args != nullptr)
  {
    arg_size = arguments->pos_args->exprs.size();
  }
  // placeholder for more complicated logic
  if(m_id->m_name == "field" && arg_size == 1)
  {
    // need to verify params, e.g., num and type
    std::string var_name = detail::strip_single_quotes(
        arguments->pos_args->exprs[0]->build_jit(n, w));
    n["field_vars"].append() = var_name;
    return var_name;
  }

  if(m_id->m_name == "mesh" && arg_size == 1)
  {
    // need to verify params, e.g., num and type
    std::string var_name = detail::strip_single_quotes(
        arguments->pos_args->exprs[0]->build_jit(n, w));
    n["mesh_vars"].append() = var_name;
    return var_name;
  }

  // mesh jit functions
  if(m_id->m_name == "volume" && arg_size == 1)
  {
    // no idea how to validate all of this,
    // but the jist is that we will add a variable
    // to the kernel of the name 'volume' so
    // it can be evaluated in the expression
    std::cout << "BOOOOOOOOOOOOOOOOM\n";
    std::string var_name = arguments->pos_args->exprs[0]->build_jit(n, w);
    n["mesh_functions"].append() = m_id->m_name;
    return m_id->m_name;
  }

  // per domain constants
  if(m_id->m_name == "domain_id" && arg_size == 0)
  {
    n["constants/domain_id/name"] = "domain_id";
    return "domain_id";
  }

  // pre-execute calls min(field('braid'))
  if((m_id->m_name == "max" || m_id->m_name == "min") && arg_size == 1)
  {
    std::cout << "MATCH\n";
    conduit::Node sub = this->build_graph(w);
    sub.print();
    std::string var_name = sub["filter_name"].as_string();
    n["pre-execute"].append() = sub;
    n["constants/" + var_name + "/name"] = var_name;
    return var_name;
  }

  // math functions supported inside the kernel
  std::string res = m_id->m_name + "(";
  for(size_t i = 0; i < arg_size; ++i)
  {
    res += arguments->pos_args->exprs[i]->build_jit(n, w);
    if(i != arg_size - 1)
      res += ",";
  }
  res += ")";

  return res;
}

conduit::Node
ASTMethodCall::build_graph(flow::Workspace &w)
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
      pos_arg_nodes[i] = arguments->pos_args->exprs[i]->build_graph(w);
      // std::cout << "flow arg :\n";
      // pos_arg_nodes[i].print();
      // std::cout << "\n";
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
      named_arg_nodes[i] = (*arguments->named_args)[i]->build_graph(w);
      named_arg_names[i] = (*arguments->named_args)[i]->key->m_name;
    }
  }

  // std::cout << "Flow method call: " << m_id->m_name << endl;

  if(!w.registry().has_entry("function_table"))
  {
    ASCENT_ERROR("Missing function table");
  }

  conduit::Node *f_table = w.registry().fetch<conduit::Node>("function_table");
  // resolve the function
  if(!f_table->has_path(m_id->m_name))
  {
    ASCENT_ERROR("unknown function " << m_id->m_name);
  }

  // resolve overloaded function names
  const conduit::Node &overload_list = (*f_table)[m_id->m_name];

  int matched_index = -1;

  std::vector<std::string> func_arg_names;
  std::unordered_set<std::string> req_args;
  std::unordered_set<std::string> opt_args;
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

    func_arg_names = func["args"].child_names();
    req_args.clear();
    opt_args.clear();
    // populate opt_args and req_args
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

    // validate positionals
    if(pos_size <= total_args)
    {
      for(int a = 0; a < pos_size; ++a)
      {
        conduit::Node func_arg = func["args"].child(a);
        // validate types
        if(func_arg["type"].as_string() != "anytype" &&
           pos_arg_nodes[a]["type"].as_string() != func_arg["type"].as_string())
        {
          valid = false;
        }
        // special case for the "scalar" pseudo-type
        if(func_arg["type"].as_string() == "scalar" &&
           detail::is_scalar(pos_arg_nodes[a]["type"].as_string()))
        {
          valid = true;
        }

        // keep track of which arguments have been specified
        if(func_arg.has_path("optional"))
        {
          valid &= opt_args.erase(func_arg_names[a]);
        }
        else
        {
          valid &= req_args.erase(func_arg_names[a]);
        }
        if(!valid)
        {
          goto next_overload;
        }
      }
    }
    else
    {
      valid = false;
    }

    // validate named arguments
    for(int a = 0; a < named_size; ++a)
    {
      // check if argument name exists
      if(!func["args"].has_path(named_arg_names[a]))
      {
        valid = false;
        goto next_overload;
      }
      // get the an argument given its name
      conduit::Node func_arg = func["args"][named_arg_names[a]];

      // validate types
      if(func_arg["type"].as_string() != "anytype" &&
         named_arg_nodes[a]["type"].as_string() != func_arg["type"].as_string())
      {
        valid = false;
      }
      // special case for the "scalar" pseudo-type
      if(func_arg["type"].as_string() == "scalar" &&
         detail::is_scalar(named_arg_nodes[a]["type"].as_string()))
      {
        valid = true;
      }

      // keep track of which arguments have been specified
      if(func_arg.has_path("optional"))
      {
        valid &= opt_args.erase(named_arg_names[a]);
      }
      else
      {
        valid &= req_args.erase(named_arg_names[a]);
      }

      if(!valid)
      {
        goto next_overload;
      }
    }

    // ensure all required arguments are passed
    if(!req_args.empty())
    {
      valid = false;
    }

    // finds the last valid function
    if(valid)
    {
      matched_index = i;
    }

  next_overload:;
  }

  conduit::Node res;

  if(matched_index != -1)
  {
    const conduit::Node &func = overload_list.child(matched_index);
    // keeps track of how things are connected
    std::map<std::string, const conduit::Node *> args_map;
    for(int a = 0; a < pos_size; ++a)
    {
      const conduit::Node &arg = pos_arg_nodes[a];
      args_map[func_arg_names[a]] = &arg;
    }
    for(int a = 0; a < named_size; ++a)
    {
      const conduit::Node &arg = named_arg_nodes[a];
      args_map[named_arg_names[a]] = &arg;
    }

    conduit::Node *subexpr_cache =
        w.registry().fetch<conduit::Node>("subexpr_cache");
    std::stringstream ess;
    ess << "method"
        << "_" << func["filter_name"].as_string() << "(";
    for(auto const &arg : args_map)
    {
      ess << arg.first << "=" << (*arg.second)["filter_name"].as_string()
          << ", ";
    }
    ess << ")";
    const std::string expr_name = ess.str();
    std::cout << expr_name << std::endl;
    if((*subexpr_cache).has_path(expr_name))
    {
      return (*subexpr_cache)[expr_name];
    }

    static int ast_method_counter = 0;
    // create a unique name for the filter
    std::stringstream ss;
    ss << "method_" << ast_method_counter++ << "_"
       << func["filter_name"].as_string();
    std::string name = ss.str();

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
    // src, dest, port

    // pass positional arguments
    for(int a = 0; a < pos_size; ++a)
    {
      const conduit::Node &arg = pos_arg_nodes[a];
      w.graph().connect(
          arg["filter_name"].as_string(), name, func_arg_names[a]);
    }

    // pass named arguments
    for(int a = 0; a < named_size; ++a)
    {
      const conduit::Node &arg = named_arg_nodes[a];
      w.graph().connect(
          arg["filter_name"].as_string(), name, named_arg_names[a]);
    }

    // connect null filter to optional args that weren't passed in
    for(std::unordered_set<std::string>::iterator it = opt_args.begin();
        it != opt_args.end();
        ++it)
    {
      w.graph().connect("null_arg", name, *it);
    }

    res["filter_name"] = name;

    // evaluate what the return type will be
    std::string res_type = func["return_type"].as_string();
    // we will need special code (per function) to determine the correct return
    // type
    if(res_type == "anytype")
    {
      // the history function's return type is the same as the type of its first
      // argument
      if(func["filter_name"].as_string() == "history")
      {
        res_type = (*args_map.at("expr_name"))["type"].as_string();
      }
      else
      {
        ASCENT_ERROR("Could not determine the return type of "
                     << func["filter_name"].as_string());
      }
    }

    res["type"] = res_type;
    (*subexpr_cache)[expr_name] = res;
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
ASTIfExpr::build_graph(flow::Workspace &w)
{
  conduit::Node n_condition = m_condition->build_graph(w);
  conduit::Node n_if = m_if->build_graph(w);
  conduit::Node n_else = m_else->build_graph(w);

  // Validate types
  const std::string condition_type = n_condition["type"].as_string();
  const std::string if_type = n_if["type"].as_string();
  const std::string else_type = n_else["type"].as_string();
  if(condition_type != "bool")
  {
    ASCENT_ERROR("if-expression condition must be of type boolean");
  }

  if(if_type != else_type)
  {
    ASCENT_ERROR("The return types of the if (" << if_type << ") and else ("
                                                << else_type
                                                << ") branches must match");
  }

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "if"
      << "_" << n_condition["filter_name"].as_string() << "_then_"
      << n_if["filter_name"].as_string() << "_else_"
      << n_else["filter_name"].as_string();
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  static int ast_if_counter = 0;
  std::stringstream ss;
  ss << "expr_if"
     << "_" << ast_if_counter++;
  std::string name = ss.str();
  conduit::Node params;
  w.graph().add_filter("expr_if", name, params);

  // src, dest, port
  w.graph().connect(n_condition["filter_name"].as_string(), name, "condition");
  w.graph().connect(n_if["filter_name"].as_string(), name, "if");
  w.graph().connect(n_else["filter_name"].as_string(), name, "else");

  conduit::Node res;
  res["type"] = if_type;
  res["filter_name"] = name;
  (*subexpr_cache)[expr_name] = res;
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
  default: std::cout << "unknown binary op " << m_op << "\n";
  }

  m_rhs->access();
  // std::cout << " op " << op_str << "\n";
  m_lhs->access();
}

bool
ASTBinaryOp::can_jit()
{
  return true;
}

std::string
ASTBinaryOp::build_jit(conduit::Node &n, flow::Workspace &w)
{
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
  default: std::cout << "unknown binary op " << m_op << "\n";
  }
  return "(" + m_lhs->build_jit(n, w) + " " + op_str + " " +
         m_rhs->build_jit(n, w) + ")";
}

conduit::Node
ASTBinaryOp::build_jit2(flow::Workspace &w)
{
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
  default: std::cout << "unknown binary op " << m_op << "\n";
  }

  conduit::Node l_in = m_lhs->build_graph(w);
  conduit::Node r_in = m_rhs->build_graph(w);

  // Validate types and evaluate what the return type will be
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();

  std::string res_type;
  if(detail::is_math(op_str))
  {
    if(detail::is_scalar(l_type) && r_type == "field" ||
       detail::is_scalar(r_type) && l_type == "field")
    {
      res_type = "field";
    }
    else
    {
      ASCENT_ERROR("Math ops are only supported on scalar and field.");
    }
  }
  else
  {
    ASCENT_ERROR("Only math ops are supported in JIT BinaryOp.");
  }

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "jit_binary_op"
      << "_" << l_in["filter_name"].as_string() << op_str
      << r_in["filter_name"].as_string();
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  static int ast_op_counter = 0;
  // create a unique name for the filter
  std::stringstream ss;
  // ss << "binary_op" << "_" << ast_op_counter << "_" << op_str;
  ss << "jit_binary_op"
     << "_" << ast_op_counter++ << "_" << m_op;
  std::string name = ss.str();

  conduit::Node params;
  params["binary_op/op_string"] = op_str;

  w.graph().add_filter("jit_filter", name, params);

  // src, dest, port
  w.graph().connect(r_in["filter_name"].as_string(), name, "arg1");
  w.graph().connect(l_in["filter_name"].as_string(), name, "arg2");

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = res_type;
  (*subexpr_cache)[expr_name] = res;
  return res;
}

conduit::Node
ASTBinaryOp::build_graph(flow::Workspace &w)
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

  conduit::Node l_in = m_lhs->build_graph(w);
  // std::cout << " flow op " << op_str << "\n";
  conduit::Node r_in = m_rhs->build_graph(w);

  // Validate types and evaluate what the return type will be
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();

  std::string res_type;
  std::stringstream msg;
  if(detail::is_math(op_str))
  {
    if((!detail::is_scalar(l_type) && l_type != "vector") ||
       (!detail::is_scalar(r_type) && r_type != "vector"))
    {
      msg << "' " << l_type << " " << op_str << " " << r_type << "'";
      ASCENT_ERROR("math operations are only supported on vectors and scalars: "
                   << msg.str());
    }

    if(detail::is_scalar(l_type) && detail::is_scalar(r_type))
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
    else
    {
      res_type = "vector";
    }
  }
  else if(detail::is_logic(op_str))
  {
    if(l_type != "bool" || r_type != "bool")
    {
      msg << "' " << l_type << " " << op_str << " " << r_type << "'";
      ASCENT_ERROR(
          "logical operators are only supported on booleans: " << msg.str());
    }
    res_type = "bool";
  }
  else
  {
    if(!detail::is_scalar(l_type) || !detail::is_scalar(r_type))
    {
      msg << "' " << l_type << " " << op_str << " " << r_type << "'";
      ASCENT_ERROR(
          "comparison operators are only supported on scalars: " << msg.str());
    }
    res_type = "bool";
  }

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "binary_op"
      << "_" << l_in["filter_name"].as_string() << op_str
      << r_in["filter_name"].as_string();
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  static int ast_op_counter = 0;
  // create a unique name for the filter
  std::stringstream ss;
  // ss << "binary_op" << "_" << ast_op_counter << "_" << op_str;
  ss << "binary_op"
     << "_" << ast_op_counter++ << "_" << m_op;
  std::string name = ss.str();

  conduit::Node params;
  params["op_string"] = op_str;

  w.graph().add_filter("expr_binary_op", name, params);

  // src, dest, port
  w.graph().connect(r_in["filter_name"].as_string(), name, "rhs");
  w.graph().connect(l_in["filter_name"].as_string(), name, "lhs");

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = res_type;
  (*subexpr_cache)[expr_name] = res;
  return res;
}

//-----------------------------------------------------------------------------
void
ASTString::access()
{
  std::cout << "Creating string " << m_name << endl;
}

std::string
ASTString::build_jit(conduit::Node &n, flow::Workspace &w)
{
  return m_name;
}

bool
ASTString::can_jit()
{
  return true;
}

conduit::Node
ASTString::build_graph(flow::Workspace &w)
{
  // strip the quotes from the variable name
  std::string stripped = detail::strip_single_quotes(m_name);

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "string"
      << "_" << stripped;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int ast_string_counter = 0;
  std::stringstream ss;
  ss << "string"
     << "_" << ast_string_counter++;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = stripped;

  w.graph().add_filter("expr_string", name, params);

  conduit::Node res;
  res["type"] = "string";
  res["filter_name"] = name;
  (*subexpr_cache)[expr_name] = res;
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
ASTBoolean::build_graph(flow::Workspace &w)
{
  // create a unique name for the filter
  static int ast_bool_counter = 0;
  std::stringstream ss;
  ss << "bool"
     << "_" << ast_bool_counter++;
  std::string name = ss.str();

  bool value = false;
  switch(tok)
  {
  case TTRUE: value = true; break;
  case TFALSE: value = false; break;
  default: std::cout << "unknown bool literal " << tok << "\n";
  }

  conduit::Node params;
  params["value"] = value;
  w.graph().add_filter("expr_bool", name, params);

  conduit::Node res;
  res["type"] = "bool";
  res["filter_name"] = name;

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
ASTArrayAccess::build_graph(flow::Workspace &w)
{
  conduit::Node n_array = array->build_graph(w);
  conduit::Node n_index = index->build_graph(w);

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

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "array_access"
      << "_" << n_array["filter_name"].as_string() << "["
      << n_index["filter_name"].as_string() << "]";
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int ast_array_counter = 0;
  std::stringstream ss;
  ss << "array_access"
     << "_" << ast_array_counter++;
  std::string name = ss.str();

  w.graph().add_filter("expr_array", name, params);

  // src, dest, port
  w.graph().connect(n_array["filter_name"].as_string(), name, "array");
  w.graph().connect(n_index["filter_name"].as_string(), name, "index");

  conduit::Node res;
  // only arrays of double are supported
  res["type"] = "double";
  res["filter_name"] = name;
  (*subexpr_cache)[expr_name] = res;
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
ASTDotAccess::build_graph(flow::Workspace &w)
{
  conduit::Node n_obj = obj->build_graph(w);

  std::string obj_type = n_obj["type"].as_string();

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
  const conduit::Node &obj = (*o_table)[obj_type];

  // resolve attribute type
  std::string path = "attrs/" + name + "/type";
  if(!obj.has_path(path))
  {
    obj.print();
    ASCENT_ERROR("Attribute " << name << " of " << obj_type << " not found");
  }
  std::string res_type = obj[path].as_string();

  conduit::Node *subexpr_cache =
      w.registry().fetch<conduit::Node>("subexpr_cache");
  std::stringstream ess;
  ess << "dot"
      << "_" << n_obj["filter_name"].as_string() << "." << name;
  const std::string expr_name = ess.str();
  if((*subexpr_cache).has_path(expr_name))
  {
    return (*subexpr_cache)[expr_name];
  }

  // create a unique name for the filter
  static int ast_dot_counter = 0;
  std::stringstream ss;
  ss << "dot"
     << "_" << ast_dot_counter++;
  std::string f_name = ss.str();

  conduit::Node params;
  params["name"] = name;

  w.graph().add_filter("expr_dot", f_name, params);

  // src, dest, port
  w.graph().connect(n_obj["filter_name"].as_string(), f_name, "obj");

  conduit::Node res;
  res["type"] = res_type;
  res["filter_name"] = f_name;
  (*subexpr_cache)[expr_name] = res;
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

bool
ASTExpressionList::can_jit()
{
  return false;
}

conduit::Node
ASTExpressionList::build_graph(flow::Workspace &w)
{
  // create a unique name for the filter
  static int ast_list_counter = 0;
  std::stringstream ss;
  ss << "list"
     << "_" << ast_list_counter++;
  std::string f_name = ss.str();

  conduit::Node params;
  w.graph().add_filter("expr_list", f_name, params);

  const size_t list_size = exprs.size();

  // Lists need to have a constant size because flow needs to know all the
  // in/out ports ahead of time.
  // We make 256 the max size.
  if(list_size > 256)
  {
    ASCENT_ERROR("Lists can have at most 256 elements.");
  }

  // Connect all items to the list
  for(size_t i = 0; i < list_size; ++i)
  {
    const conduit::Node item = exprs[i]->build_graph(w);

    std::stringstream ss;
    ss << "item" << i;

    // src, dest, port
    w.graph().connect(item["filter_name"].as_string(), f_name, ss.str());
  }

  // Fill unused items with nulls
  if(!w.graph().has_filter("null_arg"))
  {
    conduit::Node null_params;
    w.graph().add_filter("null_arg", "null_arg", null_params);
  }
  for(size_t i = list_size; i < 256; ++i)
  {
    std::stringstream ss;
    ss << "item" << i;

    // src, dest, port
    w.graph().connect("null_arg", f_name, ss.str());
  }

  conduit::Node res;
  res["type"] = "list";
  res["filter_name"] = f_name;
  return res;
}
