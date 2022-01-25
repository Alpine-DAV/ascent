//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_flow_python_interpreter.cpp
///
//-----------------------------------------------------------------------------

// always include python first
#include <Python.h>

#include "gtest/gtest.h"

#include <flow.hpp>
#include <flow_python_interpreter.hpp>


#include "t_config.hpp"

using namespace std;
using namespace flow;

//-----------------------------------------------------------------------------
TEST(flow_py_interp_exe, flow_python_interpreter)
{
    PythonInterpreter py_interp;

    EXPECT_TRUE(py_interp.initialize());
    EXPECT_TRUE(py_interp.is_running());

    EXPECT_TRUE(py_interp.run_script("print(sys.path)"));

    // test simple script, and pulling out result
    EXPECT_TRUE(py_interp.run_script("a = 42"));

    PyObject *py_a = py_interp.get_global_object("a");
    
    EXPECT_TRUE(py_a != NULL);

    int a_cpp=0;
    EXPECT_TRUE(PythonInterpreter::PyObject_to_int(py_a,a_cpp));
    EXPECT_EQ(a_cpp,42);

    // test adding val from C++ and using
    int mlt = 3;
    PyObject *py_mlt = PyLong_FromLong(mlt);
    EXPECT_TRUE(py_interp.set_global_object(py_mlt,"mlt"));

    EXPECT_TRUE(py_interp.run_script("b = a * mlt"));

    PyObject *py_b = py_interp.get_global_object("b");

    int b_cpp=0;
    EXPECT_TRUE(PythonInterpreter::PyObject_to_int(py_b,b_cpp));
    EXPECT_EQ(b_cpp,42*3);


    // test error
    EXPECT_FALSE(py_interp.run_script("badbadbad"));
    CONDUIT_INFO(py_interp.error_message());
    py_interp.clear_error();

    // test resume after error

    py_interp.run_script("print('ok')");


    // clear dict
    py_interp.reset();


    PyObject *py_a_after_clear = py_interp.get_global_object("a");
    EXPECT_TRUE(py_a_after_clear == NULL);


    // shutdown
    py_interp.shutdown();

    EXPECT_FALSE(py_interp.is_running());

}

//-----------------------------------------------------------------------------
TEST(flow_py_interp_exe, flow_python_interpreter_bad_file)
{
    PythonInterpreter py_interp;

    EXPECT_TRUE(py_interp.initialize());
    EXPECT_TRUE(py_interp.is_running());

    EXPECT_TRUE(py_interp.run_script("print(sys.path)"));
    
    EXPECT_THROW(py_interp.run_script_file("/blarg/script/path/to/garbage//thats/not/real"),
                 conduit::Error);

    // shutdown
    py_interp.shutdown();

    EXPECT_FALSE(py_interp.is_running());

}






