#include "ast.hpp"
//#include "codegen.h"
#include "parser.hpp"
#include <typeinfo>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>

using namespace std;
/* -- Code Generation -- */

void ASTInteger::access()
{
  std::cout << "Creating integer: " << m_value << endl;
}

conduit::Node ASTInteger::build_graph(flow::Workspace &w)
{
  static int ast_int_counter = 0;
  std::cout << "Flow integer: " << m_value << endl;

  // create a unique name for the filter
  std::stringstream ss;
  ss<<"integer"<<"_"<<ast_int_counter;
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
  std::cout << "Flow double: " << m_value << endl;
  static int ast_double_counter = 0;

  // create a unique name for the filter
  std::stringstream ss;
  ss<<"double"<<"_"<<ast_double_counter;
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
  std::cout << "Flow indt : " << m_name<< endl;
  conduit::Node res;
  return res;
}

void ASTMethodCall::access()
{
  const size_t size = arguments->size();
  std::cout << "Creating method call: " << m_id->m_name << endl;
  for(size_t i = 0; i < size; ++i)
  {
    std::cout<<"arg "<<i<<" = ";
    (*arguments)[i]->access();
  }
}

conduit::Node ASTMethodCall::build_graph(flow::Workspace &w)
{
  const size_t size = arguments->size();
  std::vector<conduit::Node> arg_list;
  arg_list.resize(size);
  for(size_t i = 0; i < size; ++i)
  {
    arg_list[i] = (*arguments)[i]->build_graph(w);
    std::cout<<"flow arg :\n";
    arg_list[i].print();
    std::cout<<"\n";
  }

  std::cout << "Flow method call: " << m_id->m_name << endl;

  conduit::Node res;
  return res;
}

void ASTBinaryOp::access()
{
  std::cout << "Creating binary operation " << m_op << endl;
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
    default: std::cout<<"unknown binary op "<<m_op<<"\n";

  }

  m_rhs->access();
  std::cout<<" op "<<op_str<<"\n";
  m_lhs->access();

}


conduit::Node ASTBinaryOp::build_graph(flow::Workspace &w)
{
  std::cout << "Creating binary operation " << m_op << endl;
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
    default: std::cout<<"unknown binary op "<<m_op<<"\n";

  }

  conduit::Node r_in = m_rhs->build_graph(w);
  std::cout<<" flow op "<<op_str<<"\n";
  conduit::Node l_in = m_lhs->build_graph(w);

  // Validate types
  // right now this is easy since we only have scalars and
  // mesh vars. When we have attributes(like position)
  // and vectors this validation will get more complicated
  const std::string l_type = l_in["type"].as_string();
  const std::string r_type = r_in["type"].as_string();
  if(l_type == "meshvar" || r_type == "meshvar")
  {
    std::stringstream msg;
    msg<<"' "<<l_type<<" "<<m_op<<" "<<r_type<"'";
    ASCENT_ERROR("binary operation with mesh variable not supported: "<<msg.str());
  }

  // evaluate what the return type will be
  // For now, only scalar
  std::string res_type = "scalar";

  static int ast_op_counter = 0;
  // create a unique name for the filter
  std::stringstream ss;
  //ss<<"binary_op"<<"_"<<ast_op_counter<<"_"<<op_str;
  ss<<"binary_op"<<"_"<<ast_op_counter<<"_"<<m_op;
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
  std::cout << "Creating mesh var " << m_name << endl;
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

  std::cout << "Flow mesh var " << m_name << " "<< stripped <<endl;
  conduit::Node res;

  res["type"] = "meshvar";

  //return "meshvar_" + stripped;
  return res;
}
