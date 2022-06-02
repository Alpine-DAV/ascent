//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
