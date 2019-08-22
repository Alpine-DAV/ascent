#include "ascent_expressions_ast.hpp"
//#include "codegen.h"
#include "ascent_expressions_parser.hpp"
#include <typeinfo>
#include <unordered_set>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>


using namespace std;
/* -- Code Generation -- */

namespace detail
{
std::string print_match_error(const std::string &fname,
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
    ss << ((i == 0)? "" : ", ")  << pos_arg_nodes[i]["type"].as_string();
  }
  for(int i = 0; i < named_arg_nodes.size(); ++i)
  {
    if(i == 0)
    {
      if(pos_arg_nodes.size() > 0)
      {
        ss << ", ";
      }
    }
    else
    {
      ss << ", ";
    }
    ss << named_arg_names[i] << "=" << named_arg_nodes[i]["type"].as_string();
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
      ss << n_args.child(a)["type"].as_string();
      if(a > req_args)
      {
        ss << "[optional]";
      }
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
} // namespace detail
void ASTInteger::access()
{
  std::cout << "Creating integer: " << m_value << endl;
}

conduit::Node ASTInteger::build_graph(flow::Workspace &w)
{
  static int ast_int_counter = 0;
  //std::cout << "Flow integer: " << m_value << endl;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "integer" << "_" << ast_int_counter;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_integer",
                       name,
                       params);
  ast_int_counter++;
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "scalar";
  return res;
}

void ASTDouble::access()
{
  std::cout << "Creating double: " << m_value << endl;
}

conduit::Node ASTDouble::build_graph(flow::Workspace &w)
{
  //std::cout << "Flow double: " << m_value << endl;
  static int ast_double_counter = 0;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "double" << "_" << ast_double_counter;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_value;

  w.graph().add_filter("expr_double",
                       name,
                       params);
  ast_double_counter++;

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "scalar";
  return res;
}


void ASTIdentifier::access()
{
  std::cout << "Creating identifier reference: " << m_name << endl;
  //if (context.locals().find(name) == context.locals().end())
  //{
  //  std::cerr << "undeclared variable " << name << endl;
  //  return NULL;
  //}
  //return new LoadInst(context.locals()[name], "", false, context.currentBlock());
}

conduit::Node ASTIdentifier::build_graph(flow::Workspace &w)
{
  //std::cout << "Flow indent : " << m_name << endl;
  static int ast_ident_counter = 0;

  // create a unique name for the filter
  std::stringstream ss;
  ss << "ident" << "_" << ast_ident_counter;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = m_name;

  w.graph().add_filter("expr_identifier",
                       name,
                       params);
  ast_ident_counter++;

  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = "scalar";
  return res;
}

void NamedExpression::access()
{
  key->access();
  value->access();
}

conduit::Node NamedExpression::build_graph(flow::Workspace &w)
{
  return value->build_graph(w);
}

void ASTArguments::access()
{
  if (pos_args != nullptr) {
    const size_t pos_size = pos_args->size();
    for(size_t i = 0; i < pos_size; ++i) {
        (*pos_args)[i]->access();
    }
    std::cout << "Creating positional arguments" << std::endl;
  }

  if (named_args != nullptr) {
    const size_t named_size = named_args->size();
    for(size_t i = 0; i < named_size; ++i) {
        (*named_args)[i]->access();
    }
    std::cout << "Creating named arguments" << std::endl;
  }

}

void ASTMethodCall::access()
{
  arguments->access();
  std::cout << "Creating method call: " << m_id->m_name << std::endl;
}

conduit::Node ASTMethodCall::build_graph(flow::Workspace &w)
{
  // build the positional arguments
  size_t pos_size = 0;
  std::vector<conduit::Node> pos_arg_nodes;
  if(arguments->pos_args != nullptr)
  {
    pos_size = arguments->pos_args->size();
    pos_arg_nodes.resize(pos_size);
    for(size_t i = 0; i < pos_size; ++i)
    {
      pos_arg_nodes[i] = (*arguments->pos_args)[i]->build_graph(w);
      //std::cout << "flow arg :\n";
      //pos_arg_nodes[i].print();
      //std::cout << "\n";
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

  //std::cout << "Flow method call: " << m_id->m_name << endl;

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
  std::unordered_set<std::string> all_args;
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
     all_args.clear();
    // populate opt_args, req_args, and all_args
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
      all_args.insert(func_arg_names[a]);
    }

    // validate positionals
    if(pos_size <= total_args) {
      for(int a = 0; a < pos_size; ++a)
      {
        conduit::Node func_arg = func["args"].child(a);
        // validate types
        if(pos_arg_nodes[a]["type"].as_string() != func_arg["type"].as_string())
        {
          valid = false;
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
      //check if argument name exists
      if(all_args.find(named_arg_names[a]) == all_args.end())
      {
        valid = false;
      }
      // get the an argument given its name
      conduit::Node func_arg = func["args"][named_arg_names[a]];

       // validate types
      if(named_arg_nodes[a]["type"].as_string() != func_arg["type"].as_string())
      {
        valid = false;
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
      if(named_arg_nodes[a]["type"].as_string() != func_arg["type"].as_string())
      {
        valid = false;
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
    //std::cout << "Function matched\n";
    //func.print();

    static int ast_method_counter = 0;
    // create a unique name for the filter
    std::stringstream ss;
    ss << "method_" << ast_method_counter << "_" << m_id->m_name;;
    std::string name = ss.str();
    ast_method_counter++;

    // we will have some optional parameters, prep the null_args filter
    if(!opt_args.empty())
    {
      if(!w.graph().has_filter("null_arg"))
      {
        conduit::Node null_params;
        w.graph().add_filter("null_arg",
                             "null_arg",
                             null_params);

      }
    }

    conduit::Node params;
    w.graph().add_filter(func["filter_name"].as_string(),
                         name,
                         params);

    // src, dest, port

    // pass positional arguments
    for(int a = 0; a < pos_size; ++a)
    {
      const conduit::Node &arg = pos_arg_nodes[a];
      w.graph().connect(arg["filter_name"].as_string(),name,func_arg_names[a]);
    }

    // pass named arguments
    for(int a = 0; a < named_size; ++a)
    {
      const conduit::Node &arg = named_arg_nodes[a];
      w.graph().connect(arg["filter_name"].as_string(),name,named_arg_names[a]);
    }

    // connect null filter to optional args that weren't passed in
    for(std::unordered_set<std::string>::iterator it = opt_args.begin(); it != opt_args.end(); ++it)
    {
      w.graph().connect("null_arg",name,*it);
    }

    res["filter_name"] = name;
    res["type"] = func["return_type"].as_string();
  }
  else
  {
    ASCENT_ERROR( detail::print_match_error(m_id->m_name,
                                            pos_arg_nodes,
                                            named_arg_nodes,
                                            named_arg_names,
                                            overload_list));
  }

  return res;
}

void ASTBinaryOp::access()
{
  //std::cout << "Creating binary operation " << m_op << endl;
  //Instruction::BinaryOps instr;
  std::string op_str;
  switch (m_op)
  {
    case TPLUS:   op_str = "+"; break;
    case TMINUS:  op_str = "-"; break;
    case TMUL:    op_str = "*"; break;
    case TDIV:    op_str = "/"; break;
    case TCEQ:    op_str = "=="; break;
    case TCNE:    op_str = "!="; break;
    case TCLE:    op_str = "<="; break;
    case TCGE:    op_str = ">="; break;
    case TCGT:    op_str = ">"; break;
    case TCLT:    op_str = "<"; break;
    default: std::cout << "unknown binary op " << m_op << "\n";

  }

  m_rhs->access();
  //std::cout << " op " << op_str << "\n";
  m_lhs->access();

}


conduit::Node ASTBinaryOp::build_graph(flow::Workspace &w)
{
  //std::cout << "Creating binary operation " << m_op << endl;
  std::string op_str;
  switch (m_op)
  {
    case TPLUS:   op_str = "+"; break;
    case TMINUS:  op_str = "-"; break;
    case TMUL:    op_str = "*"; break;
    case TDIV:    op_str = "/"; break;
    case TCEQ:    op_str = "=="; break;
    case TCNE:    op_str = "!="; break;
    case TCLE:    op_str = "<="; break;
    case TCGE:    op_str = ">="; break;
    case TCGT:    op_str = ">"; break;
    case TCLT:    op_str = "<"; break;
    default: std::cout << "unknown binary op " << m_op << "\n";

  }

  conduit::Node r_in = m_rhs->build_graph(w);
  //std::cout << " flow op " << op_str << "\n";
  conduit::Node l_in = m_lhs->build_graph(w);

  // Validate types
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();
  if(l_type == "meshvar" || r_type == "meshvar")
  {
    std::stringstream msg;
    msg << "' " << l_type << " " << m_op << " " << r_type << "'";
    ASCENT_ERROR("binary operation with mesh variable not supported: " << msg.str());
  }

  // evaluate what the return type will be
  // For now, only scalar

  std::string res_type = "scalar";
  if(l_type == "vector" && r_type == "vector")
  {
    res_type = "vector";
  }

  static int ast_op_counter = 0;
  // create a unique name for the filter
  std::stringstream ss;
  //ss << "binary_op" << "_" << ast_op_counter << "_" << op_str;
  ss << "binary_op" << "_" << ast_op_counter << "_" << m_op;
  std::string name = ss.str();

  conduit::Node params;
  params["op_string"] = op_str;

  w.graph().add_filter("expr_binary_op",
                       name,
                       params);

  // // src, dest, port
  w.graph().connect(r_in["filter_name"].as_string(),name,"rhs");
  w.graph().connect(l_in["filter_name"].as_string(),name,"lhs");

  ast_op_counter++;
  conduit::Node res;
  res["filter_name"] = name;
  res["type"] = res_type;
  return res;
}

void ASTMeshVar::access()
{
  //std::cout << "Creating mesh var " << m_name << endl;
}

conduit::Node ASTMeshVar::build_graph(flow::Workspace &w)
{

  // strip the quotes from the variable name
  std::string stripped = m_name;
  int pos = stripped.find("\"");
  while (pos != std::string::npos)
  {
    stripped.erase(pos,1);
    pos = stripped.find("\"");
  }

  //std::cout << "Flow mesh var " << m_name << " " << stripped << endl;
  // create a unique name for the filter
  static int ast_meshvar_counter = 0;
  std::stringstream ss;
  ss << "meshvar" << "_" << ast_meshvar_counter;
  std::string name = ss.str();

  conduit::Node params;
  params["value"] = stripped;

  w.graph().add_filter("expr_meshvar",
                       name,
                       params);
  ast_meshvar_counter++;

  conduit::Node res;
  res["type"] = "meshvar";
  res["filter_name"] = name;

  return res;
}
