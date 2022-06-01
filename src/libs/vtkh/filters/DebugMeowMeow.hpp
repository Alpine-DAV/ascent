#ifndef VTK_H_DEBUG_MEOW_MEOW_HPP
#define VTK_H_DEBUG_MEOW_MEOW_HPP

#include <vtkh/filters/util.hpp>
extern ofstream dbg;

#ifdef TRACE_DEBUG
#define DBG(x) dbg<<x
#else
#define DBG(x)
#endif

#endif
