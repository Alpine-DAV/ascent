// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: ascent_annotations.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_ANNOTATIONS_HPP
#define ASCENT_ANNOTATIONS_HPP

//-----------------------------------------------------------------------------
// -- ascent includes --
//-----------------------------------------------------------------------------
#include <ascent_logging_exports.h>
#include <ascent_logging_config.h>
#include <conduit.hpp>

#include <cstdint>
#include <string>

//-----------------------------------------------------------------------------
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//
/// ASCENT_ANNOTATE_ZZZ macros are used for caliper performance annotations.
//
//-----------------------------------------------------------------------------
#if defined(ASCENT_LOGGING_ENABLE_CALIPER)
//-----------------------------------------------------------------------------
#define ASCENT_ANNOTATE_MARK_BEGIN( name )                                     \
{                                                                              \
    std::ostringstream ascent_oss_annot;                                       \
    ascent_oss_annot << name;                                                  \
    CALI_MARK_BEGIN( ascent_oss_annot.str().c_str() );                         \
}                                                                              \
//-----------------------------------------------------------------------------
#define ASCENT_ANNOTATE_MARK_END( name )                                       \
{                                                                              \
    std::ostringstream ascent_oss_annot;                                       \
    ascent_oss_annot << name;                                                  \
    CALI_MARK_END( ascent_oss_annot.str().c_str() );                           \
}                                                                              \
//-----------------------------------------------------------------------------
#define ASCENT_ANNOTATE_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
//-----------------------------------------------------------------------------
#define ASCENT_ANNOTATE_MARK_SCOPE( name )                                     \
    std::ostringstream ascent_oss_annot;                                       \
    ascent_oss_annot << name;                                                  \
    CALI_CXX_MARK_SCOPE( ascent_oss_annot.str().c_str() );                     \

//-----------------------------------------------------------------------------
#else // these are empty when caliper is not enabled
//-----------------------------------------------------------------------------
#define ASCENT_ANNOTATE_MARK_BEGIN( name )
#define ASCENT_ANNOTATE_MARK_END( name )
#define ASCENT_ANNOTATE_MARK_FUNCTION
#define ASCENT_ANNOTATE_MARK_SCOPE( name )
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


//-----------------------------------------------------------------------------
// -- begin ascent::annotations --
//-----------------------------------------------------------------------------
namespace annotations
{
  //---------------------------------------------------------------------------
  /// Exception-safe Caliper region.
  ///
  /// The region begins in the constructor and ends in the destructor. This is
  /// a no-op when Ascent is built without Caliper support.
  //---------------------------------------------------------------------------
  class ASCENT_API Scope
  {
  public:
    explicit Scope(const std::string &name);
    ~Scope();

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    std::string m_name;
    bool m_started;
  };

  //---------------------------------------------------------------------------
  /// Stores a typed value in Caliper for the lifetime of its associated scope.
  ///
  /// Declare Value objects before the Scope they belong to if you want their
  /// values to be included when the scope ends. Values can also be set after
  /// the scope begins, which is useful when a metric is only available at the
  /// end of an operation.
  //---------------------------------------------------------------------------
  class ASCENT_API Value
  {
  public:
    explicit Value(const std::string &name);
    ~Value();

    void set(const std::string &value);
    void set(const char *value);
    void set(std::int64_t value);
    void set(std::uint64_t value);
    void set(double value);

    Value(const Value &) = delete;
    Value &operator=(const Value &) = delete;

  private:
    struct Impl;
    Impl *m_impl;
  };

  //---------------------------------------------------------------------------
  /// Caliper performance annotations environment management.
  ///
  /// Setup and tear down of Caliper.
  /// 
  /// These are all noops when Caliper is not enabled.
  ///
  /// These routines are optional, targeted for cases where caliper env vars
  /// are not used or a client code does not setup caliper itself.
  ///
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  /// Report if ascent as built with caliper support
  //---------------------------------------------------------------------------
  bool ASCENT_API supported();

  //---------------------------------------------------------------------------
  /// Initialize performance annotations
  //---------------------------------------------------------------------------
  void ASCENT_API initialize();

  //---------------------------------------------------------------------------
  /// Initialize performance annotations with options
  /// opts:
  ///   config: (optional caliper config string)
  ///   services: (optional caliper services string)
  ///   output_file: (optional string with output filename)
  //---------------------------------------------------------------------------
  void ASCENT_API initialize(const conduit::Node &opts);

  //---------------------------------------------------------------------------
  /// Flush performance annotations
  //---------------------------------------------------------------------------
  void ASCENT_API flush();

  //---------------------------------------------------------------------------
  /// Finalize performance annotations
  //---------------------------------------------------------------------------
  void ASCENT_API finalize();

}
//-----------------------------------------------------------------------------
// -- end ascent::annotations --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


#endif
