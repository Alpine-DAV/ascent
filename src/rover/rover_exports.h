#ifndef ROVER_EXPORTS_H
#define ROVER_EXPORTS_H

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms --
//-----------------------------------------------------------------------------
#if defined(_WIN32)
#if defined(ROVER_EXPORTS_FLAG)
#define ROVER_API __declspec(dllexport)
#else
#define ROVER_API __declspec(dllimport)
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
# if __GNUC__ >= 4 && defined(ROVER_EXPORTS_FLAG)
#   define ROVER_API __attribute__ ((visibility("default")))
# else
#   define ROVER_API /* hidden by default */
# endif
#endif

#endif



