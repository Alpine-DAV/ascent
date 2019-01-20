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

void ASTDouble::access()
{
  std::cout << "Creating double: " << m_value << endl;
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

void ASTMethodCall::access()
{
  //Function *function = context.module->getFunction(id.name.c_str());
  //if (function == NULL)
  //{
  //  std::cerr << "no such function " << id.name << endl;
  //}
  std::vector<void*> args;
  ExpressionList::const_iterator it;
  for (it = arguments.begin(); it != arguments.end(); it++)
  {
    //args.push_back((**it).codeGen(context));
    (**it).access();
  }
  //CallInst *call = CallInst::Create(function, makeArrayRef(args), "", context.currentBlock());
  std::cout << "Creating method call: " << m_id.m_name << endl;
  //return call;
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

  m_rhs.access();
  std::cout<<" op "<<op_str<<"\n";
  m_lhs.access();

//  return NULL;
//math:
//  return BinaryOperator::Create(instr, lhs.codeGen(context),
//    rhs.codeGen(context), "", context.currentBlock());
}

void ASTMeshVar::access()
{
  std::cout << "Creating mesh var " << m_name << endl;
//  if (context.locals().find(lhs.name) == context.locals().end())
//  {
//    std::cerr << "undeclared variable " << lhs.name << endl;
//    return NULL;
//  }
//  return new StoreInst(rhs.codeGen(context), context.locals()[lhs.name], false, context.currentBlock());
}
