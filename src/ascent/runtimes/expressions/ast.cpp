#include "ast.hpp"
//#include "codegen.h"
#include "parser.hpp"
#include <typeinfo>
using namespace std;
/* -- Code Generation -- */

void ASTInteger::access()
{
  std::cout << "Creating integer: " << m_value << endl;
}

std::string ASTInteger::build_graph(flow::Workspace &w)
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
  return name;
}

void ASTDouble::access()
{
  std::cout << "Creating double: " << m_value << endl;
}

std::string ASTDouble::build_graph(flow::Workspace &w)
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
  return name;
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

std::string ASTIdentifier::build_graph(flow::Workspace &w)
{
  std::cout << "Flow indt : " << m_name<< endl;
  return "";
}

void ASTMethodCall::access()
{
  const size_t size = arguments->size();
  for(size_t i = 0; i < size; ++i)
  {
    (*arguments)[i]->access();
  }
  std::cout << "Creating method call: " << m_id->m_name << endl;
}

std::string ASTMethodCall::build_graph(flow::Workspace &w)
{
  const size_t size = arguments->size();
  for(size_t i = 0; i < size; ++i)
  {
    (*arguments)[i]->access();
  }

  std::cout << "Flow method call: " << m_id->m_name << endl;
  return "";
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


std::string ASTBinaryOp::build_graph(flow::Workspace &w)
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

  std::string r_in = m_rhs->build_graph(w);
  std::cout<<" flow op "<<op_str<<"\n";
  std::string l_in  = m_lhs->build_graph(w);

  static int ast_op_counter = 0;
  // create a unique name for the filter
  std::stringstream ss;
  ss<<"binary_op"<<"_"<<ast_op_counter<<"_"<<op_str;
  std::string name = ss.str();

  conduit::Node params;
  params["op_string"] = op_str;

  w.graph().add_filter("expr_binary_op",
                       name,
                       params);

  // // src, dest, port
  w.graph().connect(r_in,name,"rhs");
  w.graph().connect(l_in,name,"lhs");

  ast_op_counter++;
  return name;
}
void ASTMeshVar::access()
{
  std::cout << "Creating mesh var " << m_name << endl;
}

std::string ASTMeshVar::build_graph(flow::Workspace &w)
{
  std::cout << "Flow mesh var " << m_name << endl;
  return "";
}
