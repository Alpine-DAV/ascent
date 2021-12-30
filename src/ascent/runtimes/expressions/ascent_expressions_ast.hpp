//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_RUNTIME_AST
#define ASCENT_RUNTIME_AST

#include "ascent_derived_jit.hpp"
#include "flow_workspace.hpp"
#include <iostream>
#include <vector>

// alias because it's too long
namespace expressions = ascent::runtime::expressions;

class ASTVisitor;

class ASTNode
{
public:
  virtual ~ASTNode()
  {
  }
  virtual void accept(ASTVisitor *visitor) const = 0;
};

class ASTExpression : public ASTNode
{
public:
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTIdentifier : public ASTExpression
{
public:
  std::string m_name;
  ASTIdentifier(const std::string &name) : m_name(name)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTNamedExpression : public ASTExpression
{
public:
  ASTIdentifier *key;
  ASTExpression *value;
  ASTNamedExpression(ASTIdentifier *key, ASTExpression *value)
      : key(key), value(value)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTNamedExpression()
  {
    delete key;
    delete value;
  }
};

using ASTNamedExpressionList = std::vector<ASTNamedExpression *>;

class ASTBlock : public ASTNode
{
public:
  ASTNamedExpressionList *stmts;
  ASTExpression *expr;
  ASTBlock(ASTNamedExpressionList *stmts, ASTExpression *expr)
      : stmts(stmts), expr(expr)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTBlock()
  {
    delete expr;
    for(auto stmt : *stmts)
    {
      delete stmt;
    }
    stmts->clear();
    delete stmts;
  }
};

class ASTInteger : public ASTExpression
{
public:
  int m_value;
  ASTInteger(int value) : m_value(value)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTDouble : public ASTExpression
{
public:
  double m_value;
  ASTDouble(double value) : m_value(value)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTString : public ASTExpression
{
public:
  std::string m_name;
  ASTString(const std::string &name) : m_name(name)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTBoolean : public ASTExpression
{
public:
  int tok;
  ASTBoolean(const int tok) : tok(tok)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;
};

class ASTExpressionList : public ASTExpression
{
public:
  std::vector<ASTExpression *> exprs;
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTExpressionList()
  {
    for(size_t i = 0; i < exprs.size(); ++i)
    {
      delete exprs[i];
    }
  }
};

class ASTArguments
{
public:
  ASTExpressionList *pos_args;
  ASTNamedExpressionList *named_args;
  ASTArguments(ASTExpressionList *pos_args, ASTNamedExpressionList *named_args)
      : pos_args(pos_args), named_args(named_args)
  {
  }

  virtual ~ASTArguments()
  {
    delete pos_args;

    if(named_args != nullptr)
    {
      const size_t named_size = named_args->size();
      for(size_t i = 0; i < named_size; ++i)
      {
        delete(*named_args)[i];
      }
      delete named_args;
    }
  }
};

class ASTMethodCall : public ASTExpression
{
public:
  ASTIdentifier *m_id;
  ASTArguments *arguments;
  ASTMethodCall(ASTIdentifier *id, ASTArguments *arguments)
      : m_id(id), arguments(arguments)
  {
  }
  ASTMethodCall(ASTIdentifier *id) : m_id(id)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTMethodCall()
  {
    delete arguments;
    delete m_id;
  }
};

class ASTBinaryOp : public ASTExpression
{
public:
  ASTExpression *m_lhs;
  int m_op;
  ASTExpression *m_rhs;
  ASTBinaryOp(ASTExpression *lhs, int op, ASTExpression *rhs)
      : m_lhs(lhs), m_op(op), m_rhs(rhs)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTBinaryOp()
  {
    delete m_lhs;
    delete m_rhs;
  }
};

class ASTIfExpr : public ASTExpression
{
public:
  ASTExpression *m_condition;
  ASTExpression *m_if;
  ASTExpression *m_else;
  ASTIfExpr(ASTExpression *m_condition,
            ASTExpression *m_if,
            ASTExpression *m_else)
      : m_condition(m_condition), m_if(m_if), m_else(m_else)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTIfExpr()
  {
    delete m_condition;
    delete m_if;
    delete m_else;
  }
};

class ASTArrayAccess : public ASTExpression
{
public:
  ASTExpression *array;
  ASTExpression *index;
  ASTArrayAccess(ASTExpression *array, ASTExpression *index)
      : array(array), index(index)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTArrayAccess()
  {
    delete array;
    delete index;
  }
};

class ASTDotAccess : public ASTExpression
{
public:
  ASTExpression *obj;
  std::string name;
  ASTDotAccess(ASTExpression *obj, const std::string &name)
      : obj(obj), name(name)
  {
  }
  virtual void accept(ASTVisitor *visitor) const override;

  virtual ~ASTDotAccess()
  {
    delete obj;
  }
};

// abstract base class for visiting ASTs using the visitor pattern
class ASTVisitor
{
public:
  virtual ~ASTVisitor()
  {
  }
  virtual void visit(const ASTBlock &block) = 0;
  virtual void visit(const ASTExpression &expr) = 0;
  virtual void visit(const ASTInteger &expr) = 0;
  virtual void visit(const ASTDouble &expr) = 0;
  virtual void visit(const ASTIdentifier &expr) = 0;
  virtual void visit(const ASTNamedExpression &expr) = 0;
  virtual void visit(const ASTMethodCall &call) = 0;
  virtual void visit(const ASTIfExpr &expr) = 0;
  virtual void visit(const ASTBinaryOp &expr) = 0;
  virtual void visit(const ASTString &expr) = 0;
  virtual void visit(const ASTBoolean &expr) = 0;
  virtual void visit(const ASTArrayAccess &expr) = 0;
  virtual void visit(const ASTDotAccess &expr) = 0;
  virtual void visit(const ASTExpressionList &list) = 0;
};

class PrintVisitor final : public ASTVisitor
{
public:
  void visit(const ASTBlock &block) override;
  void visit(const ASTExpression &expr) override;
  void visit(const ASTInteger &expr) override;
  void visit(const ASTDouble &expr) override;
  void visit(const ASTIdentifier &expr) override;
  void visit(const ASTNamedExpression &expr) override;
  void visit(const ASTMethodCall &call) override;
  void visit(const ASTIfExpr &expr) override;
  void visit(const ASTBinaryOp &expr) override;
  void visit(const ASTString &expr) override;
  void visit(const ASTBoolean &expr) override;
  void visit(const ASTArrayAccess &expr) override;
  void visit(const ASTDotAccess &expr) override;
  void visit(const ASTExpressionList &list) override;
};

class BuildGraphVisitor final : public ASTVisitor
{
public:
  BuildGraphVisitor(
      flow::Workspace &w,
      const std::shared_ptr<const expressions::JitExecutionPolicy> exec_policy,
      const bool verbose);

  void visit(const ASTBlock &block) override;
  void visit(const ASTExpression &expr) override;
  void visit(const ASTInteger &expr) override;
  void visit(const ASTDouble &expr) override;
  void visit(const ASTIdentifier &expr) override;
  void visit(const ASTNamedExpression &expr) override;
  void visit(const ASTMethodCall &call) override;
  void visit(const ASTIfExpr &expr) override;
  void visit(const ASTBinaryOp &expr) override;
  void visit(const ASTString &expr) override;
  void visit(const ASTBoolean &expr) override;
  void visit(const ASTArrayAccess &expr) override;
  void visit(const ASTDotAccess &expr) override;
  void visit(const ASTExpressionList &list) override;

  conduit::Node get_output() const
  {
    return output;
  }

  conduit::Node table() const
  {
    return symbol_table;
    //return subexpr_cache;
  }

private:
  flow::Workspace &w;
  const bool verbose;
  conduit::Node output;
  // used to eliminate common subexpressions
  // ex: (x - min) / (max - min) then min should only be evaluated once
  conduit::Node subexpr_cache;
  int ast_counter = 0;
  const std::shared_ptr<const expressions::JitExecutionPolicy> exec_policy;
  // we only have one scope so no need for a stack of symbol tables
  conduit::Node symbol_table;
};

#endif
