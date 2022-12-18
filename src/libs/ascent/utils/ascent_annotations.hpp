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
#include <ascent_exports.h>
#include <ascent_config.h>
#include <conduit.hpp>

//-----------------------------------------------------------------------------
#if defined(ASCENT_CALIPER_ENABLED)
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//
/// ASCENT_ANNOTATE_ZZZ macros are used for caliper performance annotations.
//
//-----------------------------------------------------------------------------
#if defined(ASCENT_CALIPER_ENABLED)
#define ASCENT_ANNOTATE_MARK_BEGIN( name ) CALI_MARK_BEGIN( name )
#define ASCENT_ANNOTATE_MARK_END( name ) CALI_MARK_END( name )
#define ASCENT_ANNOTATE_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define ASCENT_ANNOTATE_MARK_SCOPE( name )  CALI_CXX_MARK_SCOPE( name )
#else // these are empty when caliper is not enabled
#define ASCENT_ANNOTATE_MARK_BEGIN( name )
#define ASCENT_ANNOTATE_MARK_END( name )
#define ASCENT_ANNOTATE_MARK_FUNCTION
#define ASCENT_ANNOTATE_MARK_SCOPE( name )
#endif

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
