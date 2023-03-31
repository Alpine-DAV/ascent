//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_expression_eval.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_EXPRESSION_EVAL_HPP
#define ASCENT_EXPRESSION_EVAL_HPP
#include <conduit.hpp>
#include <ascent_exports.h>
#include <ascent_data_object.hpp>

#include "flow_workspace.hpp"
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

namespace expressions
{

void ASCENT_API register_builtin();
void ASCENT_API initialize_functions();
void ASCENT_API initialize_objects();

struct Cache
{
  conduit::Node m_data;
  int m_rank;
  bool m_filtered = false;
  bool m_loaded = false;
  std::string m_session_file;

  void load(const std::string &dir,
            const std::string &session);

  double last_known_time();
  void last_known_time(double time);
  void filter_time(double ftime);
  bool filtered();
  bool loaded();
  void save();
  // allow saving with an alternative name
  void save(const std::string &filename);
  void save(const std::string &filename,
            const std::vector<std::string> &selection);

  ~Cache();
};

static conduit::Node m_function_table;

class ASCENT_API ExpressionEval
{
protected:
  DataObject m_data_object;
  flow::Workspace w;
  static Cache m_cache;
  void jit_root(conduit::Node &root, const std::string &expr_name);
public:
  ExpressionEval(DataObject &dataset);
  ExpressionEval(conduit::Node *dataset);
  DataObject& data_object();

  static const conduit::Node &get_cache();
  static void get_last(conduit::Node &data);
  static void reset_cache();
  static void load_cache(const std::string &dir,
                         const std::string &session);

  // helpers for saving cache files
  static void save_cache(const std::string &filename,
                         const std::vector<std::string> &selection);
  static void save_cache(const std::string &filename);
  static void save_cache();

  conduit::Node evaluate(const std::string expr, std::string exp_name = "");
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::expressions--
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

