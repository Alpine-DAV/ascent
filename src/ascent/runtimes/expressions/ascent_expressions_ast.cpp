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
// -- Accept Methods
//-----------------------------------------------------------------------------
//{{{
void
ASTExpression::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTInteger::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTDouble::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTIdentifier::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTNamedExpression::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTMethodCall::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTIfExpr::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTArrayAccess::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTDotAccess::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTBoolean::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTBinaryOp::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTString::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
void
ASTExpressionList::accept(ASTVisitor *visitor) const
{
  visitor->visit(*this);
}
//}}}

//-----------------------------------------------------------------------------
// -- PrintVisitor
//-----------------------------------------------------------------------------
//{{{
void
PrintVisitor::visit(const ASTExpression &expr)
{
  std::cout << "placeholder expression" << std::endl;
}

void
PrintVisitor::visit(const ASTInteger &expr)
{
  std::cout << "Creating integer: " << expr.m_value << endl;
}

void
PrintVisitor::visit(const ASTDouble &expr)
{
  std::cout << "Creating double: " << expr.m_value << endl;
}

void
PrintVisitor::visit(const ASTIdentifier &expr)
{
  std::cout << "Creating identifier reference: " << expr.m_name << endl;
}

void
PrintVisitor::visit(const ASTNamedExpression &expr)
{
  expr.key->accept(this);
  expr.value->accept(this);
}

void
PrintVisitor::visit(const ASTMethodCall &call)
{
  std::cout << "Creating method call: " << call.m_id->m_name << std::endl;
  const ASTArguments &args = *call.arguments;
  if(args.pos_args != nullptr)
  {
    std::cout << "Creating positional arguments" << std::endl;
    const size_t pos_size = args.pos_args->exprs.size();
    for(size_t i = 0; i < pos_size; ++i)
    {
      args.pos_args->exprs[i]->accept(this);
    }
  }

  if(args.named_args != nullptr)
  {
    std::cout << "Creating named arguments" << std::endl;
    const size_t named_size = args.named_args->size();
    for(size_t i = 0; i < named_size; ++i)
    {
      (*args.named_args)[i]->accept(this);
    }
  }
}

void
PrintVisitor::visit(const ASTIfExpr &expr)
{
  std::cout << "Creating if expression" << std::endl;

  std::cout << "Creating if condition" << std::endl;
  expr.m_condition->accept(this);

  std::cout << "Creating if body" << std::endl;
  expr.m_if->accept(this);

  std::cout << "Creating else body" << std::endl;
  expr.m_else->accept(this);
}

void
PrintVisitor::visit(const ASTBinaryOp &expr)
{
  // std::cout << "Creating binary operation " << m_op << endl;
  // Instruction::BinaryOps instr;
  std::string op_str;
  switch(expr.m_op)
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
  default: ASCENT_ERROR("unknown binary op " << expr.m_op);
  }

  expr.m_lhs->accept(this);
  std::cout << " op " << op_str << "\n";
  expr.m_rhs->accept(this);
}

void
PrintVisitor::visit(const ASTString &expr)
{
  std::cout << "Creating string " << expr.m_name << endl;
}

void
PrintVisitor::visit(const ASTBoolean &expr)
{
  std::string bool_str = "";
  switch(expr.tok)
  {
  case TTRUE: bool_str = "True"; break;
  case TFALSE: bool_str = "False"; break;
  default: std::cout << "unknown bool literal " << expr.tok << "\n";
  }
  std::cout << "Creating bool literal " << bool_str << std::endl;
}

void
PrintVisitor::visit(const ASTArrayAccess &expr)
{
  std::cout << "Creating array access" << std::endl;

  std::cout << "Creating array" << std::endl;
  expr.array->accept(this);

  std::cout << "Creating array index" << std::endl;
  expr.index->accept(this);
}

void
PrintVisitor::visit(const ASTDotAccess &expr)
{
  std::cout << "Creating dot access" << std::endl;

  std::cout << "Creating object" << std::endl;
  expr.obj->accept(this);

  std::cout << "Creating dot name " << expr.name << std::endl;
}

void
PrintVisitor::visit(const ASTExpressionList &list)
{
  std::cout << "Creating list" << std::endl;
  for(auto expr : list.exprs)
  {
    expr->accept(this);
  }
}
//}}}

//-----------------------------------------------------------------------------
// -- BuildGraphVisitor
//-----------------------------------------------------------------------------
//{{{

BuildGraphVisitor::BuildGraphVisitor(flow::Workspace &w, const bool verbose)
    : w(w), verbose(verbose)
{
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTExpression &expr)
{
  // placeholder to make binary op work with "not"
  if(!w.graph().has_filter("bop_placeholder"))
  {
    conduit::Node placeholder_params;
    placeholder_params["value"] = true;
    w.graph().add_filter("expr_bool", "bop_placeholder", placeholder_params);
  }
  output["filter_name"] = "bop_placeholder";
  output["type"] = "bool";
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTInteger &expr)
{
  // create a unique name for each expression so we can reuse subexpressions
  std::stringstream ss;
  ss << "integer_" << expr.m_value;
  const std::string verbose_name = ss.str();
  if(subexpr_cache.has_path(verbose_name))
  {
    output = subexpr_cache[verbose_name];
    return;
  }

  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    std::stringstream ss;
    ss << "integer_" << ast_counter++;
    name = ss.str();
  }

  conduit::Node params;
  params["value"] = expr.m_value;

  w.graph().add_filter("expr_integer", name, params);
  output["filter_name"] = name;
  output["type"] = "int";
  subexpr_cache[verbose_name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTDouble &expr)
{
  std::stringstream ss;
  ss << "double_" << expr.m_value;
  const std::string verbose_name = ss.str();
  if(subexpr_cache.has_path(verbose_name))
  {
    output = subexpr_cache[verbose_name];
    return;
  }

  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    std::stringstream ss;
    ss << "double_" << ast_counter++;
    name = ss.str();
  }

  conduit::Node params;
  params["value"] = expr.m_value;

  w.graph().add_filter("expr_double", name, params);

  output["filter_name"] = name;
  output["type"] = "double";
  subexpr_cache[verbose_name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTIdentifier &expr)
{
  std::stringstream ss;
  ss << "ident_" << expr.m_name;
  const std::string name = ss.str();
  if(subexpr_cache.has_path(name))
  {
    output = subexpr_cache[name];
    return;
  }

  // get identifier type from cache
  conduit::Node *cache = w.registry().fetch<conduit::Node>("cache");
  if(!cache->has_path(expr.m_name))
  {
    ASCENT_ERROR("Unknown expression identifier: '" << expr.m_name << "'");
  }

  const int entries = (*cache)[expr.m_name].number_of_children();
  if(entries < 1)
  {
    ASCENT_ERROR("Expression identifier: needs a non-zero number of entries: "
                 << entries);
  }

  conduit::Node params;
  params["value"] = expr.m_name;

  w.graph().add_filter("expr_identifier", name, params);

  output["filter_name"] = name;
  // grab the last one calculated
  output["type"] = (*cache)[expr.m_name].child(entries - 1)["type"];
  subexpr_cache[name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTNamedExpression &expr)
{
  return expr.value->accept(this);
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTMethodCall &call)
{
  // visit the positional arguments
  size_t pos_size = 0;
  std::vector<conduit::Node> pos_arg_nodes;
  if(call.arguments->pos_args != nullptr)
  {
    pos_size = call.arguments->pos_args->exprs.size();
    pos_arg_nodes.resize(pos_size);
    for(size_t i = 0; i < pos_size; ++i)
    {
      call.arguments->pos_args->exprs[i]->accept(this);
      pos_arg_nodes[i] = output;
    }
  }

  // build the named arguments
  size_t named_size = 0;
  std::vector<conduit::Node> named_arg_nodes;
  // stores argument names in the same order as named_arg_nodes
  std::vector<std::string> named_arg_names;
  if(call.arguments->named_args != nullptr)
  {
    named_size = call.arguments->named_args->size();
    named_arg_nodes.resize(named_size);
    named_arg_names.resize(named_size);
    for(size_t i = 0; i < named_size; ++i)
    {
      (*call.arguments->named_args)[i]->accept(this);
      named_arg_nodes[i] = output;
      named_arg_names[i] = (*call.arguments->named_args)[i]->key->m_name;
    }
  }

  if(!w.registry().has_entry("function_table"))
  {
    ASCENT_ERROR("Missing function table");
  }

  conduit::Node *f_table = w.registry().fetch<conduit::Node>("function_table");
  // resolve the function
  if(!f_table->has_path(call.m_id->m_name))
  {
    ASCENT_ERROR("Expressions: Unknown function '" << call.m_id->m_name
                                                   << "'.");
  }

  // resolve overloaded function names
  const conduit::Node &overload_list = (*f_table)[call.m_id->m_name];

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

  if(matched_index != -1)
  {
    const conduit::Node &func = overload_list.child(matched_index);

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
      if(subexpr_cache.has_path(verbose_name))
      {
        output = subexpr_cache[verbose_name];
        return;
      }
      if(verbose)
      {
        name = verbose_name;
      }
      else
      {
        std::stringstream ss;
        ss << "jit_method_" << ast_counter++ << "_"
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
      if(subexpr_cache.has_path(verbose_name))
      {
        output = subexpr_cache[verbose_name];
        return;
      }
      if(verbose)
      {
        name = verbose_name;
      }
      else
      {
        std::stringstream ss;
        ss << "method_" << ast_counter++ << "_"
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
          if(!subexpr_cache.has_path(jit_execute_name))
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
    std::string output_type = func["return_type"].as_string();
    // we will need special code (per function) to determine the correct return
    // type
    if(output_type == "anytype")
    {
      // the history function's return type is the type of its first argument
      if(func["filter_name"].as_string() == "history")
      {
        output_type = (*args_map["expr_name"])["type"].as_string();
      }
      else
      {
        ASCENT_ERROR("Could not determine the return type of "
                     << func["filter_name"].as_string());
      }
    }

    output["filter_name"] = name;
    output["type"] = output_type;
    subexpr_cache[verbose_name] = output;
  }
  else
  {
    ASCENT_ERROR(detail::print_match_error(call.m_id->m_name,
                                           pos_arg_nodes,
                                           named_arg_nodes,
                                           named_arg_names,
                                           overload_list));
  }
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTIfExpr &expr)
{
  expr.m_condition->accept(this);
  conduit::Node n_condition = output;
  expr.m_if->accept(this);
  conduit::Node n_if = output;
  expr.m_else->accept(this);
  conduit::Node n_else = output;

  // Validate types
  const std::string condition_type = n_condition["type"].as_string();
  const std::string if_type = n_if["type"].as_string();
  const std::string else_type = n_else["type"].as_string();
  std::string output_type;
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
    output_type = "jitable";
  }
  else if(condition_type == "bool")
  {
    if(if_type != else_type)
    {
      ASCENT_ERROR("The return types of the if (" << if_type << ") and else ("
                                                  << else_type
                                                  << ") branches must match");
    }
    output_type = if_type;
  }
  else
  {
    ASCENT_ERROR("if-expression condition is of type: '"
                 << condition_type
                 << "' but must be of type 'bool' or a field type.");
  }

  std::string name;
  std::string verbose_name;
  if(output_type == "jitable")
  {
    std::stringstream ss;
    ss << "jit_if_" << n_condition["filter_name"].as_string() << "_then_"
       << n_if["filter_name"].as_string() << "_else_"
       << n_else["filter_name"].as_string();
    verbose_name = ss.str();
    if(subexpr_cache.has_path(verbose_name))
    {
      output = subexpr_cache[verbose_name];
      return;
    }
    if(verbose)
    {
      name = verbose_name;
    }
    else
    {
      std::stringstream ss;
      ss << "jit_if_" << ast_counter++;
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
    if(subexpr_cache.has_path(verbose_name))
    {
      output = subexpr_cache[verbose_name];
      return;
    }
    if(verbose)
    {
      name = verbose_name;
    }
    else
    {
      std::stringstream ss;
      ss << "if_" << ast_counter++;
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
  output["filter_name"] = name;
  output["type"] = output_type;
  subexpr_cache[verbose_name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTBinaryOp &expr)
{
  // std::cout << "Creating binary operation " << m_op << endl;
  std::string op_str;
  switch(expr.m_op)
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
  default: ASCENT_ERROR("unknown binary op " << expr.m_op);
  }
  expr.m_lhs->accept(this);
  conduit::Node l_in = output;
  expr.m_rhs->accept(this);
  conduit::Node r_in = output;

  std::stringstream ss;
  // flow doesn't like it when we have a / in the filter name
  ss << "binary_op"
     << "(" << l_in["filter_name"].as_string()
     << (op_str == "/" ? "div" : op_str) << r_in["filter_name"].as_string()
     << ")";
  const std::string verbose_name = ss.str();
  if(subexpr_cache.has_path(verbose_name))
  {
    output = subexpr_cache[verbose_name];
  }
  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    std::stringstream ss;
    ss << "binary_op_" << ast_counter++;
    name = ss.str();
  }

  // Validate types and evaluate what the return type will be
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();

  std::string output_type;
  if(detail::is_math(op_str))
  {
    if((detail::is_scalar(l_type) && detail::is_field_type(r_type)) ||
       (detail::is_scalar(r_type) && detail::is_field_type(l_type)) ||
       (detail::is_field_type(l_type) && detail::is_field_type(r_type)))
    {
      output_type = "jitable";
    }
    else if(detail::is_scalar(l_type) && detail::is_scalar(r_type))
    {
      // promote to double if at least one is a double
      if(l_type == "double" || r_type == "double")
      {
        output_type = "double";
      }
      else
      {
        output_type = "int";
      }
    }
    else if(l_type == "vector" && r_type == "vector")
    {
      output_type = "vector";
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
      output_type = "jitable";
    }
    else if(l_type != "bool" || r_type != "bool")
    {
      ASCENT_ERROR(
          "logical operators are only supported on bools and field types: "
          << "'" << l_type << " " << op_str << " " << r_type << "'");
    }
    else
    {
      output_type = "bool";
    }
  }
  else
  {
    // comparison ops
    if((detail::is_scalar(l_type) && detail::is_field_type(r_type)) ||
       (detail::is_scalar(r_type) && detail::is_field_type(l_type)) ||
       (detail::is_field_type(l_type) && detail::is_field_type(r_type)))
    {
      output_type = "jitable";
    }
    else if(!detail::is_scalar(l_type) || !detail::is_scalar(r_type))
    {
      ASCENT_ERROR(
          "comparison operators are only supported on scalars and field types: "
          << "'" << l_type << " " << op_str << " " << r_type << "'");
    }
    else
    {
      output_type = "bool";
    }
  }

  if(output_type == "jitable")
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
  output["filter_name"] = name;
  output["type"] = output_type;
  subexpr_cache[verbose_name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTString &expr)
{
  // strip the quotes from the variable name
  std::string stripped = detail::strip_single_quotes(expr.m_name);

  std::stringstream ss;
  ss << "string_" << stripped;
  const std::string name = ss.str();
  if(subexpr_cache.has_path(name))
  {
    output = subexpr_cache[name];
    return;
  }

  conduit::Node params;
  params["value"] = stripped;

  w.graph().add_filter("expr_string", name, params);

  output["filter_name"] = name;
  output["type"] = "string";
  subexpr_cache[name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTBoolean &expr)
{
  bool value = false;
  switch(expr.tok)
  {
  case TTRUE: value = true; break;
  case TFALSE: value = false; break;
  default: std::cout << "unknown bool literal " << expr.tok << "\n";
  }

  std::stringstream ss;
  ss << "bool_" << value;
  const std::string name = ss.str();
  if(subexpr_cache.has_path(name))
  {
    output = subexpr_cache[name];
    return;
  }

  conduit::Node params;
  params["value"] = value;
  w.graph().add_filter("expr_bool", name, params);

  output["filter_name"] = name;
  output["type"] = "bool";
  subexpr_cache[name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTArrayAccess &expr)
{
  expr.array->accept(this);
  conduit::Node n_array = output;
  expr.index->accept(this);
  conduit::Node n_index = output;

  std::stringstream ss;
  ss << "array_access_" << n_array["filter_name"].as_string() << "["
     << n_index["filter_name"].as_string() << "]";
  const std::string name = ss.str();
  if(subexpr_cache.has_path(name))
  {
    output = subexpr_cache[name];
    return;
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

  // only arrays of double are supported
  output["filter_name"] = name;
  output["type"] = "double";
  subexpr_cache[name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTDotAccess &expr)
{
  expr.obj->accept(this);
  const conduit::Node input_obj = output;

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
  std::string path = "attrs/" + expr.name + "/type";
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
    ASCENT_ERROR("Attribute " << expr.name << " of " << obj_type
                              << " not found." << ss.str());
  }
  std::string output_type = o_table_obj[path].as_string();

  std::string f_name;
  std::string verbose_name;
  if(o_table->has_path(expr.name + "/jitable") || output_type == "jitable")
  {
    std::stringstream ss;
    ss << "jit_dot_(" << input_obj["filter_name"].as_string() << ")__"
       << expr.name;
    verbose_name = ss.str();
    if(subexpr_cache.has_path(verbose_name))
    {
      output = subexpr_cache[verbose_name];
      return;
    }
    if(verbose)
    {
      f_name = verbose_name;
    }
    else
    {
      std::stringstream ss;
      ss << "jit_dot_" << ast_counter++ << "__" << expr.name;
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
    params["name"] = expr.name;
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
    ss << "dot_(" << input_obj["filter_name"].as_string() << ")__" << expr.name;
    verbose_name = ss.str();
    if(subexpr_cache.has_path(verbose_name))
    {
      output = subexpr_cache[verbose_name];
      return;
    }
    if(verbose)
    {
      f_name = verbose_name;
    }
    else
    {
      std::stringstream ss;
      ss << "dot_" << ast_counter++ << "__" << expr.name;
      f_name = ss.str();
    }

    conduit::Node params;
    params["name"] = expr.name;

    w.graph().add_filter("expr_dot", f_name, params);
    // src, dest, port
    w.graph().connect(input_obj["filter_name"].as_string(), f_name, "obj");
  }

  output["filter_name"] = f_name;
  output["type"] = output_type;
  subexpr_cache[verbose_name] = output;
}

//-----------------------------------------------------------------------------
void
BuildGraphVisitor::visit(const ASTExpressionList &list)
{
  const size_t list_size = list.exprs.size();

  std::stringstream ss;
  ss << "list_[";
  std::vector<conduit::Node> items;
  for(size_t i = 0; i < list_size; ++i)
  {
    list.exprs[i]->accept(this);
    items.push_back(output);
    ss << items.back()["filter_name"].as_string();
    if(i < list_size - 1)
    {
      ss << ", ";
    }
  }
  ss << "]";
  const std::string verbose_name = ss.str();
  if(subexpr_cache.has_path(verbose_name))
  {
    subexpr_cache[verbose_name];
  }
  std::string name;
  if(verbose)
  {
    name = verbose_name;
  }
  else
  {
    std::stringstream ss;
    ss << "list_" << ast_counter++;
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

  output["filter_name"] = name;
  output["type"] = "list";
  subexpr_cache[verbose_name] = output;
}
//}}}
