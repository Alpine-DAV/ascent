//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_vtkh_device_utils.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_vtkh_device_utils.hpp"
#include "ascent/runtimes/expressions/ascent_array.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>
#include <type_traits>


// VTKm includes
#define VTKM_USE_DOUBLE_PRECISION


using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin detail:: --
//-----------------------------------------------------------------------------
namespace detail
{

};
//-----------------------------------------------------------------------------
// -- end detail:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// VTKHDataAdapter device methods
//-----------------------------------------------------------------------------

// Definition of the cast function with __host__ __device__
ASCENT_EXEC void 
VTKHDeviceAdapter::castUint64ToFloat64(const uint64_t* input, double* output, int size) {
    Array<double> int_to_double(input, size);
    output = int_to_double.get_ptr(memory_space);

    // init device array
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        output[i] = static_cast<double>(input[i]);
    });
    ASCENT_DEVICE_ERROR_CHECK();
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
