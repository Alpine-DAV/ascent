//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_derived_jit.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_DERVIVED_JIT_HPP
#define ASCENT_DERVIVED_JIT_HPP

#include <ascent.hpp>
#include <conduit.hpp>
#include <flow.hpp>
#include <memory>

#include "ascent_jit_array.hpp"
#include "ascent_jit_field.hpp"
#include "ascent_jit_kernel.hpp"
#include "ascent_jit_math.hpp"
#include "ascent_jit_topology.hpp"
#include "ascent_insertion_ordered_set.hpp"
// Matt: there is a lot of code that needs its own file

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

class Jitable
{
protected:
  static int m_cuda_device_id;
public:
  Jitable(const int num_domains)
  {
    for(int i = 0; i < num_domains; ++i)
    {
      dom_info.append();
    }
    arrays.resize(num_domains);
  }

  static void init_occa();
  static void set_cuda_device(int device_id);
  static int num_cuda_devices();


  void fuse_vars(const Jitable &from);
  bool can_execute() const;
  void execute(conduit::Node &dataset, const std::string &field_name);
  std::string generate_kernel(const int dom_idx,
                              const conduit::Node &args) const;

  // map of kernel types (e.g. for different topologies)
  std::unordered_map<std::string, Kernel> kernels;
  // stores entries and argument values for each domain
  conduit::Node dom_info;
  // Store the array schemas. Used by code generation. We will copy to these
  // schemas when we execute
  std::vector<ArrayCode> arrays;
  std::string topology;
  std::string association;
  // metadata used to make the . operator work and store various jitable state
  conduit::Node obj;
};

class MemoryRegion
{
public:
  MemoryRegion(const void *start, const void *end);
  MemoryRegion(const void *start, const size_t size);
  bool operator<(const MemoryRegion &other) const;

  const unsigned char *start;
  const unsigned char *end;
  mutable bool allocated;
  mutable size_t index;
};

class JitExecutionPolicy
{
public:
  JitExecutionPolicy();
  virtual bool should_execute(const Jitable &jitable) const = 0;
  virtual std::string get_name() const = 0;
};

class FusePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class AlwaysExecutePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class RoundtripPolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

// fuse until the number of bytes in args exceeds a threshold
class InputBytesPolicy final : public JitExecutionPolicy
{
public:
  InputBytesPolicy(const size_t num_bytes);
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;

private:
  const size_t num_bytes;
};

void pack_topology(const std::string &topo_name,
                   const conduit::Node &domain,
                   conduit::Node &args,
                   ArrayCode &array);
void pack_array(const conduit::Node &array,
                const std::string &name,
                conduit::Node &args,
                ArrayCode &array_code);
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
