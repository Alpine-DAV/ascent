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
/// file: ascent_jit_math.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_math.hpp"
#include <ascent_logging.hpp>

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

void
MathCode::determinant_2x2(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &res_name,
                          const bool declare) const
{
  code.insert((declare ? "const double " : "") + res_name + " = " + a +
              "[0] * " + b + "[1] - " + b + "[0] * " + a + "[1];\n");
}
void
MathCode::determinant_3x3(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &c,
                          const std::string &res_name,
                          const bool declare) const
{
  code.insert((declare ? "const double " : "") + res_name + " = " + a +
              "[0] * (" + b + "[1] * " + c + "[2] - " + c + "[1] * " + b +
              "[2]) - " + a + "[1] * (" + b + "[0] * " + c + "[2] - " + c +
              "[0] * " + b + "[2]) + " + a + "[2] * (" + b + "[0] * " + c +
              "[1] - " + c + "[0] * " + b + "[1]);\n");
}

void
MathCode::vector_subtract(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &res_name,
                          const int num_components,
                          const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_components) +
                "];\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] - " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::vector_add(InsertionOrderedSet<std::string> &code,
                     const std::string &a,
                     const std::string &b,
                     const std::string &res_name,
                     const int num_components,
                     const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_components) +
                "];\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] + " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::cross_product(InsertionOrderedSet<std::string> &code,
                        const std::string &a,
                        const std::string &b,
                        const std::string &res_name,
                        const int num_components,
                        const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[3];\n");
  }
  if(num_components == 3)
  {
    code.insert(res_name + "[0] = " + a + "[1] * " + b + "[2] - " + a +
                "[2] * " + b + "[1];\n");
    code.insert(res_name + "[1] = " + a + "[2] * " + b + "[0] - " + a +
                "[0] * " + b + "[2];\n");
  }
  else
  {
    ASCENT_ERROR("cross_product is not implemented for vectors '"
                 << a << "' and '" << b << "' with " << num_components
                 << "components.");
  }
  code.insert(res_name + "[2] = " + a + "[0] * " + b + "[1] - " + a + "[1] * " +
              b + "[0];\n");
}

void
MathCode::dot_product(InsertionOrderedSet<std::string> &code,
                      const std::string &a,
                      const std::string &b,
                      const std::string &res_name,
                      const int num_components,
                      const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + " = 0.0;\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + " += " + a + "[" +
                std::to_string(i) + "] * " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::magnitude(InsertionOrderedSet<std::string> &code,
                    const std::string &a,
                    const std::string &res_name,
                    const int num_components,
                    const bool declare) const
{
  if(num_components == 3)
  {
    code.insert((declare ? "const double " : "") + res_name + " = sqrt(" + a +
                "[0] * " + a + "[0] + " + a + "[1] * " + a + "[1] + " + a +
                "[2] * " + a + "[2]);\n");
  }
  else if(num_components == 2)
  {
    code.insert((declare ? "const double " : "") + res_name + " = sqrt(" + a +
                "[0] * " + a + "[0] + " + a + "[1] * " + a + "[1]);\n");
  }
  else
  {
    ASCENT_ERROR("magnitude for vector '" << a << "' of size " << num_components
                                          << " is not implemented.");
  }
}

void
MathCode::array_avg(InsertionOrderedSet<std::string> &code,
                    const int length,
                    const std::string &array_name,
                    const std::string &res_name,
                    const bool declare) const
{
  std::stringstream array_avg;
  array_avg << "(";
  for(int i = 0; i < length; ++i)
  {
    if(i != 0)
    {
      array_avg << " + ";
    }
    array_avg << array_name + "[" << i << "]";
  }
  array_avg << ") / " << length;
  code.insert((declare ? "const double " : "") + res_name + " = " +
              array_avg.str() + ";\n");
}

// average value of a component given an array of vectors
void
MathCode::component_avg(InsertionOrderedSet<std::string> &code,
                        const int length,
                        const std::string &array_name,
                        const std::string &coord,
                        const std::string &res_name,
                        const bool declare) const
{
  const int component = coord[0] - 'x';
  std::stringstream comp_avg;
  comp_avg << "(";
  for(int i = 0; i < length; ++i)
  {
    if(i != 0)
    {
      comp_avg << " + ";
    }
    comp_avg << array_name + "[" << i << "][" << component << "]";
  }
  comp_avg << ") / " << length;
  code.insert((declare ? "const double " : "") + res_name + " = " +
              comp_avg.str() + ";\n");
}

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
