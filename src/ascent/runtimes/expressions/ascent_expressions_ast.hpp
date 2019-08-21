#ifndef ASCENT_RUNTIME_AST
#define ASCENT_RUNTIME_AST
#include <iostream>
#include <vector>
#include "flow_workspace.hpp"

class ASTExpression;

typedef std::vector<ASTExpression*> ExpressionList;

class ASTNode {
public:
  virtual ~ASTNode() {}
  virtual void access() {}
  virtual conduit::Node build_graph(flow::Workspace &w)
                                     { 
                                         conduit::Node res;
                                         return res;
                                     }
};

class ASTExpression : public ASTNode {
};

class ASTInteger : public ASTExpression {
public:
  int m_value;
  ASTInteger(int value) : m_value(value) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class ASTDouble : public ASTExpression {
public:
  double m_value;
  ASTDouble(double value) : m_value(value) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class ASTIdentifier : public ASTExpression {
public:
  std::string m_name;
  ASTIdentifier(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class ASTMeshVar: public ASTExpression
{
public:
  std::string m_name;
  ASTMeshVar(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class ASTMethodCall : public ASTExpression {
public:
  ASTIdentifier *m_id;
  ExpressionList *arguments;
  ASTMethodCall(ASTIdentifier *id, ExpressionList *arguments) :
    m_id(id), arguments(arguments) { }
  ASTMethodCall(ASTIdentifier *id) : m_id(id) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~ASTMethodCall()
  {
    const size_t size = arguments->size();
    for(size_t i = 0; i < size; ++i)
    {
      delete (*arguments)[i];
    }
    delete arguments;
    delete m_id;
  }
};

class ASTBinaryOp : public ASTExpression {
public:
  int m_op;
  ASTExpression *m_lhs;
  ASTExpression *m_rhs;
  ASTBinaryOp(ASTExpression *lhs, int op, ASTExpression *rhs) :
    m_lhs(lhs), m_rhs(rhs), m_op(op) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual ~ASTBinaryOp()
  {
    delete m_lhs;
    delete m_rhs;
  }
};
#endif
