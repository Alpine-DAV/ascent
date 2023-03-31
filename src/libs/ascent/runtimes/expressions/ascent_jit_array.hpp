//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_array.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_JIT_ARRAY_HPP
#define ASCENT_JIT_ARRAY_HPP

#include <unordered_map>
#include <string>
#include <conduit.hpp>

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

// codegen_arrays are a way to trick the codegen into bundling multiple fields
// into a single vector field by constructing a false schema. Their components
// are accessed using just the "component_name" rather than
// "<array_name>_<component_name>". They are not packed into args.
class SchemaBool
{
public:
  SchemaBool(const conduit::Schema &schema, bool codegen_array)
      : schema(schema), codegen_array(codegen_array){};

  conduit::Schema schema;
  bool codegen_array;
};

// num_components should be 0 if you don't want an mcarray
void
schemaFactory(const std::string &schema_type,
              const conduit::DataType::TypeID type_id,
              const size_t component_size,
              const int num_components,
              conduit::Schema &out_schema);

void
schemaFactory(const std::string &schema_type,
              const conduit::DataType::TypeID type_id,
              const size_t component_size,
              const std::vector<std::string> &component_names,
              conduit::Schema &out_schema);

class ArrayCode
{
public:
  std::string index(const std::string &array_name,
                    const std::string &idx,
                    const int component = -1) const;

  std::string index(const std::string &idx,
                    const std::string name,
                    const std::ptrdiff_t offset,
                    const std::ptrdiff_t stride,
                    const size_t pointer_size) const;

  std::string index(const std::string &array_name,
                    const std::string &idx,
                    const std::string &component) const;

  std::unordered_map<std::string, SchemaBool> array_map;
};
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

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
