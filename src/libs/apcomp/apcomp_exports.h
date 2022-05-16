//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: apcomp_exports.h
///
//-----------------------------------------------------------------------------

#ifndef APCOMP_EXPORTS_H
#define APCOMP_EXPORTS_H

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms --
//-----------------------------------------------------------------------------
#if defined(_WIN32)
#if defined(ASCENT_EXPORTS_FLAG)
#define APCOMP_API __declspec(dllexport)
#else
#define APCOMP_API __declspec(dllimport)
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
# if __GNUC__ >= 4 && defined(ASCENT_EXPORTS_FLAG)
#   define APCOMP_API __attribute__ ((visibility("default")))
# else
#   define APCOMP_API /* hidden by default */
# endif
#endif

#endif



