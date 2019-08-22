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
  virtual void access() = 0;
  virtual conduit::Node build_graph(flow::Workspace &w) = 0;
};

class ASTExpression : public ASTNode {
};

class ASTIdentifier : public ASTExpression {
public:
  std::string m_name;
  ASTIdentifier(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class NamedExpression : public ASTExpression {
public:
  ASTIdentifier *key;
  ASTExpression *value;
  NamedExpression(ASTIdentifier *key, ASTExpression *value) : key(key), value(value) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~NamedExpression() {
    delete key;
    delete value;
  }
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

class ASTMeshVar: public ASTExpression {
public:
  std::string m_name;
  ASTMeshVar(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

typedef std::vector<ASTExpression*> ExpressionList;
typedef std::vector<NamedExpression*> NamedExpressionList;

class ASTArguments {
public:
  ExpressionList *pos_args;
  NamedExpressionList *named_args;
  ASTArguments(ExpressionList *pos_args, NamedExpressionList *named_args) :
    pos_args(pos_args), named_args(named_args) { }
  virtual void access();

  virtual ~ASTArguments() {
    if (pos_args != nullptr) {
      const size_t pos_size = pos_args->size();
      for(size_t i = 0; i < pos_size; ++i)
      {
        delete (*pos_args)[i];
      }
      delete pos_args;
    }

    if (named_args != nullptr) {
      const size_t named_size = named_args->size();
      for(size_t i = 0; i < named_size; ++i)
      {
        delete (*named_args)[i];
      }
      delete named_args;
    }
  }
};

class ASTMethodCall : public ASTExpression {
public:
  ASTIdentifier *m_id;
  ASTArguments *arguments;
  ASTMethodCall(ASTIdentifier *id, ASTArguments *arguments) :
    m_id(id), arguments(arguments) { }
  ASTMethodCall(ASTIdentifier *id) : m_id(id) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~ASTMethodCall()
  {
    delete arguments;
    delete m_id;
  }
};

class ASTBinaryOp : public ASTExpression {
public:
  int m_op;
  ASTExpression *m_lhs;
  ASTExpression *m_rhs;
  ASTBinaryOp(ASTExpression* lhs, int op, ASTExpression* rhs) :
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
