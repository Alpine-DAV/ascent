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
/// file: ascent_jit_kernel.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_kernel.hpp"
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
Kernel::fuse_kernel(const Kernel &from)
{
  functions.insert(from.functions);
  kernel_body.insert(from.kernel_body);
  for_body.insert(from.for_body);
}

// copy expr into a variable (scalar or vector) "output"
std::string
Kernel::generate_output(const std::string &output, bool declare) const
{
  std::string res;
  if(declare)
  {
    res += "double " + output;
    if(num_components > 1)
    {
      res += "[" + std::to_string(num_components) + "]";
    }
    res += ";\n";
  }
  if(num_components > 1)
  {
    for(int i = 0; i < num_components; ++i)
    {
      res += output + "[" + std::to_string(i) + "] = " + expr + "[" +
             std::to_string(i) + "];\n";
    }
  }
  else
  {
    res += output + " = " + expr + ";\n";
  }
  return res;
}

// generate a loop to set expr into the array "output"
std::string
Kernel::generate_loop(const std::string &output,
                      const ArrayCode &array_code,
                      const std::string &entries_name) const
{
  // clang-format off
  std::string res =
    "for (int group = 0; group < " + entries_name + "; group += 128; @outer)\n"
       "{\n"
         "for (int item = group; item < (group + 128); ++item; @inner)\n"
         "{\n"
           "if (item < " + entries_name + ")\n"
           "{\n" +
              for_body.accumulate();
              if(num_components > 1)
              {
                for(int i = 0; i < num_components; ++i)
                {
                  res += array_code.index(output, "item", i) + " = " + expr +
                    "[" + std::to_string(i) + "];\n";
                }
              }
              else
              {
                res += array_code.index(output, "item") + " = " + expr + ";\n";
              }
  res +=
           "}\n"
         "}\n"
       "}\n";
  return res;
  // clang-format on
}

//-----------------------------------------------------------------------------
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
