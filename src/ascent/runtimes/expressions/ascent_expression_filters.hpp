//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_expression_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_EXPRESSION_FILTERS
#define ASCENT_EXPRESSION_FILTERS

#include <ascent.hpp>

#include <flow_filter.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

//-----------------------------------------------------------------------------
///
/// Filters for expressions
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class NullArg : public ::flow::Filter
{
public:
  NullArg();
  ~NullArg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Identifier : public ::flow::Filter
{
public:
  Identifier();
  ~Identifier();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class History : public ::flow::Filter
{
public:
  History();
  ~History();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Boolean : public ::flow::Filter
{
public:
  Boolean();
  ~Boolean();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Integer : public ::flow::Filter
{
public:
  Integer();
  ~Integer();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Double : public ::flow::Filter
{
public:
  Double();
  ~Double();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class String : public ::flow::Filter
{
public:
  String();
  ~String();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinaryOp : public ::flow::Filter
{
public:
  BinaryOp();
  ~BinaryOp();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Cycle : public ::flow::Filter
{
public:
  Cycle();
  ~Cycle();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ScalarMax : public ::flow::Filter
{
public:
  ScalarMax();
  ~ScalarMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ScalarMin : public ::flow::Filter
{
public:
  ScalarMin();
  ~ScalarMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayMin : public ::flow::Filter
{
public:
  ArrayMin();
  ~ArrayMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldMax : public ::flow::Filter
{
public:
  FieldMax();
  ~FieldMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayMax : public ::flow::Filter
{
public:
  ArrayMax();
  ~ArrayMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldMin : public ::flow::Filter
{
public:
  FieldMin();
  ~FieldMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldAvg : public ::flow::Filter
{
public:
  FieldAvg();
  ~FieldAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayAvg : public ::flow::Filter
{
public:
  ArrayAvg();
  ~ArrayAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldSum : public ::flow::Filter
{
public:
  FieldSum();
  ~FieldSum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArraySum : public ::flow::Filter
{
public:
  ArraySum();
  ~ArraySum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldNanCount : public ::flow::Filter
{
public:
  FieldNanCount();
  ~FieldNanCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldInfCount : public ::flow::Filter
{
public:
  FieldInfCount();
  ~FieldInfCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Vector : public ::flow::Filter
{
public:
  Vector();
  ~Vector();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Magnitude : public ::flow::Filter
{
public:
  Magnitude();
  ~Magnitude();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Axis : public ::flow::Filter
{
public:
  Axis();
  ~Axis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Histogram : public ::flow::Filter
{
public:
  Histogram();
  ~Histogram();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Binning : public ::flow::Filter
{
public:
  Binning();
  ~Binning();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Entropy : public ::flow::Filter
{
public:
  Entropy();
  ~Entropy();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Pdf : public ::flow::Filter
{
public:
  Pdf();
  ~Pdf();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Cdf : public ::flow::Filter
{
public:
  Cdf();
  ~Cdf();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class IfExpr : public ::flow::Filter
{
public:
  IfExpr();
  ~IfExpr();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Field : public ::flow::Filter
{
public:
  Field();
  ~Field();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinByIndex : public ::flow::Filter
{
public:
  BinByIndex();
  ~BinByIndex();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinByValue : public ::flow::Filter
{
public:
  BinByValue();
  ~BinByValue();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayAccess : public ::flow::Filter
{
public:
  ArrayAccess();
  ~ArrayAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class DotAccess : public ::flow::Filter
{
public:
  DotAccess();
  ~DotAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ExpressionList : public ::flow::Filter
{
public:
  ExpressionList();
  ~ExpressionList();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Quantile : public ::flow::Filter
{
public:
  Quantile();
  ~Quantile();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class PointAndAxis : public ::flow::Filter
{
public:
  PointAndAxis();
  ~PointAndAxis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class MaxFromPoint : public ::flow::Filter
{
public:
  MaxFromPoint();
  ~MaxFromPoint();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Bin : public ::flow::Filter
{
public:
  Bin();
  ~Bin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Bounds : public ::flow::Filter
{
public:
  Bounds();
  ~Bounds();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
