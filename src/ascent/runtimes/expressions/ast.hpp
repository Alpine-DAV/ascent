#include <iostream>
#include <vector>

class ASTExpression;

typedef std::vector<ASTExpression*> ExpressionList;

class ASTNode {
public:
  virtual ~ASTNode() {}
  virtual void access() {}
};

class ASTExpression : public ASTNode {
};

class ASTInteger : public ASTExpression {
public:
  int m_value;
  ASTInteger(int value) : m_value(value) { }
  virtual void access();
};

class ASTDouble : public ASTExpression {
public:
  double m_value;
  ASTDouble(double value) : m_value(value) { }
  virtual void access();
};

class ASTIdentifier : public ASTExpression {
public:
  std::string m_name;
  ASTIdentifier(const std::string& name) : m_name(name) { }
  virtual void access();
};

class ASTMeshVar: public ASTExpression
{
public:
  std::string m_name;
  ASTMeshVar(const std::string& name) : m_name(name) { }
  virtual void access();
};

class ASTMethodCall : public ASTExpression {
public:
  const ASTIdentifier& m_id;
  ExpressionList arguments;
  ASTMethodCall(const ASTIdentifier& id, ExpressionList& arguments) :
    m_id(id), arguments(arguments) { }
  ASTMethodCall(const ASTIdentifier& id) : m_id(id) { }
  virtual void access();
};

class ASTBinaryOp : public ASTExpression {
public:
  int m_op;
  ASTExpression& m_lhs;
  ASTExpression& m_rhs;
  ASTBinaryOp(ASTExpression& lhs, int op, ASTExpression& rhs) :
    m_lhs(lhs), m_rhs(rhs), m_op(op) { }
  virtual void access();
};
