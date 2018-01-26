//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://ascent.readthedocs.io/.
// 
// Please also read ascent/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
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




