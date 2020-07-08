#include "ascent_expressions_ast.hpp"
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

conduit::Node
ASTInteger::build_graph(flow::Workspace &w)
{
  static int ast_int_counter = 0;
  // std::cout << "Flow integer: " << m_value << endl;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "integer"
     << "_" << ast_int_counter++;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_integer", name, params);
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "int";
  return res;
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
  return res;
}

//-----------------------------------------------------------------------------
void
ASTIdentifier::access()
{
  std::cout << "Creating identifier reference: " << m_name << endl;
  // if (context.locals().find(name) == context.locals().end())
  //{
  //  std::cerr << "undeclared variable " << name << endl;
  //  return NULL;
  //}
  // return new LoadInst(context.locals()[name], "", false,
  // context.currentBlock());
}

conduit::Node
ASTIdentifier::build_graph(flow::Workspace &w)
{
  // std::cout << "Flow indent : " << m_name << endl;
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
ASTNamedExpression::build_graph(flow::Workspace &w)
{
  return value->build_graph(w);
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
    // std::cout << "Function matched\n";
    // func.print();

    static int ast_method_counter = 0;
    // create a unique name for the filter
    std::stringstream ss;
    ss << "method_" << ast_method_counter++ << "_" << m_id->m_name;
    ;
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

    // keeps track of how things are connected
    std::unordered_map<std::string, const conduit::Node *> args_map;

    // pass positional arguments
    for(int a = 0; a < pos_size; ++a)
    {
      const conduit::Node &arg = pos_arg_nodes[a];
      w.graph().connect(
          arg["filter_name"].as_string(), name, func_arg_names[a]);

      args_map[func_arg_names[a]] = &arg;
    }

    // pass named arguments
    for(int a = 0; a < named_size; ++a)
    {
      const conduit::Node &arg = named_arg_nodes[a];
      w.graph().connect(
          arg["filter_name"].as_string(), name, named_arg_names[a]);

      args_map[named_arg_names[a]] = &arg;
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

  m_lhs->access();
  // std::cout << " op " << op_str << "\n";
  m_rhs->access();
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
  return res;
}

//-----------------------------------------------------------------------------
void
ASTString::access()
{
  std::cout << "Creating string " << m_name << endl;
}

conduit::Node
ASTString::build_graph(flow::Workspace &w)
{
  // strip the quotes from the variable name
  std::string stripped = m_name;
  int pos = stripped.find("'");
  while(pos != std::string::npos)
  {
    stripped.erase(pos, 1);
    pos = stripped.find("'");
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

  // create a unique name for the filter
  static int ast_array_counter = 0;
  std::stringstream ss;
  ss << "array"
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
