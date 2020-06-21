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
  virtual std::string build_string(conduit::Node &n)
  {
    return "banana";
  };
};

class ASTExpression : public ASTNode {
public:
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
};

class ASTIdentifier : public ASTExpression {
public:
  std::string m_name;
  ASTIdentifier(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual std::string build_string(conduit::Node &n);
};

class NamedExpression : public ASTExpression {
public:
  ASTIdentifier *key;
  ASTExpression *value;
  NamedExpression(ASTIdentifier *key, ASTExpression *value) : key(key), value(value) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual std::string build_string(conduit::Node &n);

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
  virtual std::string build_string(conduit::Node &n);
};

class ASTDouble : public ASTExpression {
public:
  double m_value;
  ASTDouble(double value) : m_value(value) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual std::string build_string(conduit::Node &n);
};

class ASTString: public ASTExpression {
public:
  std::string m_name;
  ASTString(const std::string& name) : m_name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual std::string build_string(conduit::Node &n);
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
  virtual std::string build_string(conduit::Node &n);

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
  ASTBinaryOp(ASTExpression *lhs, int op, ASTExpression *rhs) :
    m_lhs(lhs), m_rhs(rhs), m_op(op) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);
  virtual std::string build_string(conduit::Node &n);

  virtual ~ASTBinaryOp()
  {
    delete m_lhs;
    delete m_rhs;
  }
};

class ASTIfExpr : public ASTExpression {
public:
  ASTExpression *m_condition;
  ASTExpression *m_if;
  ASTExpression *m_else;
  ASTIfExpr(ASTExpression *m_condition, ASTExpression *m_if, ASTExpression *m_else) :
    m_condition(m_condition), m_if(m_if), m_else(m_else) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~ASTIfExpr()
  {
    delete m_condition;
    delete m_if;
    delete m_else;
  }
};


class ASTArrayAccess : public ASTExpression {
public:
  ASTExpression *array;
  ASTExpression *index;
  ASTArrayAccess(ASTExpression *array, ASTExpression *index) :
    array(array), index(index) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~ASTArrayAccess()
  {
    delete array;
    delete index;
  }
};

class ASTDotAccess : public ASTExpression {
public:
  ASTExpression *obj;
  std::string name;
  ASTDotAccess(ASTExpression *obj, const std::string& name) :
    obj(obj), name(name) { }
  virtual void access();
  virtual conduit::Node build_graph(flow::Workspace &w);

  virtual ~ASTDotAccess()
  {
    delete obj;
  }
};
#endif
