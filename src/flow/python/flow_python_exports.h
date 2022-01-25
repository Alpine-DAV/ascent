//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: flow_python_exports.h
///
//-----------------------------------------------------------------------------

#ifndef FLOW_PYTHON_EXPORTS_H
#define FLOW_PYTHON_EXPORTS_H

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms --
//-----------------------------------------------------------------------------

#if defined(_WIN32)
#if defined(FLOW_PYTHON_EXPORTS) || defined(flow_python_EXPORTS)
#define FLOW_PYTHON_API __declspec(dllexport)
#else
#define FLOW_PYTHON_API __declspec(dllimport)
#endif
#if defined(_MSC_VER)
// Turn off warning about lack of DLL interface
#pragma warning(disable:4251)
// Turn off warning non-dll class is base for dll-interface class.
#pragma warning(disable:4275)
// Turn off warning about identifier truncation
#pragma warning(disable:4786)
#endif
#else
# if __GNUC__ >= 4 && (defined(FLOW_PYTHON_EXPORTS) || defined(flow_python_EXPORTS))
#   define FLOW_PYTHON_API __attribute__ ((visibility("default")))
# else
#   define FLOW_PYTHON_API /* hidden by default */
# endif
#endif

#endif



