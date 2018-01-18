//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: flow_python_interpreter.cpp
///
//-----------------------------------------------------------------------------

#include "flow_python_interpreter.hpp"

// standard lib includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

using namespace std;

//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
///
/// PythonInterpreter Constructor
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
PythonInterpreter::PythonInterpreter()
{
    m_handled_init = false;
    m_running      = false;
    m_error        = false;
    
    m_py_main_module = NULL;
    m_py_global_dict = NULL;
    
    m_py_trace_module = NULL;
    m_py_sio_module = NULL;
    m_py_trace_print_exception_func = NULL;
    m_py_sio_class = NULL;
    
}

//-----------------------------------------------------------------------------
///
/// PythonInterpreter Destructor
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
PythonInterpreter::~PythonInterpreter()
{
    // Shutdown the interpreter if running.
    shutdown();
}

//-----------------------------------------------------------------------------
///
/// Starts the python interpreter. If no arguments are passed creates
/// suitable dummy arguments
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::initialize(int argc, char **argv)
{
    // if already running, ignore
    if(m_running)
        return true;

    // Check Py_IsInitialized(), some one else may have inited python
    if(Py_IsInitialized())
    {
        // make sure we know we don't need to clean up the interp
        m_handled_init = false;
    }
    else
    {
        // set prog name
        char *prog_name = (char*)"flow_embedded_py";
        
        if(argc == 0 || argv == NULL)
        {
            Py_SetProgramName(prog_name);
        }
        else
        {
            Py_SetProgramName(argv[0]);
        }

        // Init Python
        Py_Initialize();
        PyEval_InitThreads();

        // set sys argvs
        
        if(argc == 0 || argv == NULL)
        {
            PySys_SetArgv(1, &prog_name);
        }
        else
        {
            PySys_SetArgv(argc, argv);
        }
        
        // make sure we know we need to cleanup the interp
        m_handled_init = true;
    }

    
    // do to setup b/c we need for c++ connection , even if python was already
    // inited

    // setup up __main__ and capture StdErr
    PyRun_SimpleString("import os,sys,traceback,StringIO\n");
    if(check_error())
        return false;

    // all of these PyObject*s are borrowed refs
    m_py_main_module = PyImport_AddModule((char*)"__main__");
    m_py_global_dict = PyModule_GetDict(m_py_main_module);

    // get objects that help us print an exception.

    // get ref to traceback.print_exception method
    m_py_trace_module = PyImport_AddModule("traceback");
    PyObject *py_trace_dict = PyModule_GetDict(m_py_trace_module);
    m_py_trace_print_exception_func = PyDict_GetItemString(py_trace_dict,
                                                           "print_exception");
    // get ref to StringIO.StringIO class
    m_py_sio_module   = PyImport_AddModule("StringIO");
    PyObject *py_sio_dict= PyModule_GetDict(m_py_sio_module);
    m_py_sio_class = PyDict_GetItemString(py_sio_dict,"StringIO");
    
    m_running = true;

    return true;
}


//-----------------------------------------------------------------------------
///
/// Resets the state of the interpreter if it is running
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
void
PythonInterpreter::reset()
{
    if(m_running)
    {
        // clean gloal dict.
        PyDict_Clear(m_py_global_dict);
    }
}

//-----------------------------------------------------------------------------
///
/// Shuts down the interpreter if it is running
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
void
PythonInterpreter::shutdown()
{
    if(m_running)
    {
        if(m_handled_init)
        {
            Py_Finalize();
        }
        
        m_running = false;
        m_handled_init = false;
    }
}


//-----------------------------------------------------------------------------
///
/// Adds passed path to "sys.path"
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::add_system_path(const std::string &path)
{
    return run_script("sys.path.insert(1,r'" + path + "')\n");
}

//-----------------------------------------------------------------------------
///
/// Executes passed python script in the interpreter
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::run_script(const std::string &script)
{
    bool res = false;
    if(m_running)
    {
        CONDUIT_INFO("PythonInterpreter::run_script " << script);
        PyRun_String((char*)script.c_str(),
                     Py_file_input,
                     m_py_global_dict,
                     m_py_global_dict);
        if(!check_error())
            res = true;
    }
    return res;
}

//-----------------------------------------------------------------------------
///
/// Executes passed python script in the interpreter
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::run_script_file(const std::string &fname)
{
    ifstream ifs(fname.c_str());
    string py_script((istreambuf_iterator<char>(ifs)),
                     istreambuf_iterator<char>());
    ifs.close();
    return run_script(py_script);
}

//-----------------------------------------------------------------------------
///
/// Adds C python object to the global dictionary.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::set_global_object(PyObject *py_obj,
                                     const string &py_name)
{
    PyDict_SetItemString(m_py_global_dict, py_name.c_str(), py_obj);
    return !check_error();
}

//-----------------------------------------------------------------------------
///
/// Get C python object from the global dictionary.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
PyObject *
PythonInterpreter::get_global_object(const string &py_name)
{
    PyObject *res = PyDict_GetItemString(m_py_global_dict, py_name.c_str());
    if(check_error())
        res = NULL;
    return res;
}


//-----------------------------------------------------------------------------
///
/// Checks python error state and constructs appropriate error message
/// if an error did occur. It can be used to check for errors in both
/// python scripts & calls to the C-API. The difference between these
/// to cases is the existence of a python traceback.
///
/// Note: This method clears the python error state, but it will continue 
/// to return "true" indicating an error until clear_error() is called.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool 
PythonInterpreter::check_error()
{
    if(PyErr_Occurred())
    {
        m_error = true;
        m_error_msg = "<Unknown Error>";

        string sval ="";
        PyObject *py_etype;
        PyObject *py_eval;
        PyObject *py_etrace;

        PyErr_Fetch(&py_etype, &py_eval, &py_etrace);
        
        if(py_etype)
        {
            PyErr_NormalizeException(&py_etype, &py_eval, &py_etrace);

            if(PyObject_to_string(py_etype, sval))
            {
                m_error_msg = sval;
            }

            if(py_eval)
            {
                if(PyObject_to_string(py_eval, sval))
                {
                    m_error_msg += sval;
                }
            }
            
            if(py_etrace)
            {
                if(PyTraceback_to_string(py_etype, py_eval, py_etrace, sval))
                {
                    m_error_msg += "\n" + sval;
                }
            }
        }
        
        PyErr_Restore(py_etype, py_eval, py_etrace);
        PyErr_Clear();
    }

    return m_error;
}

//-----------------------------------------------------------------------------
///
/// Clears environment error flag and message.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
void 
PythonInterpreter::clear_error()
{
    if(m_error)
    {
        m_error = false;
        m_error_msg = "";
    }
}

//-----------------------------------------------------------------------------
///
/// Helper that converts a python object to a double.
/// Returns true if the conversion succeeds.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::PyObject_to_double(PyObject *py_obj, double &res)
{
    if(PyFloat_Check(py_obj))
    {
        res = PyFloat_AS_DOUBLE(py_obj);
        return true;
    }

    if(PyInt_Check(py_obj))
    {
        res = (double) PyInt_AS_LONG(py_obj);
        return true;
    }

    if(PyLong_Check(py_obj))
    {
        res = PyLong_AsDouble(py_obj);
        return true;
    }

    if(PyNumber_Check(py_obj) != 1)
        return false;

    PyObject *py_val = PyNumber_Float(py_obj);
    if(py_val == NULL)
        return false;
    res = PyFloat_AS_DOUBLE(py_val);
    Py_DECREF(py_val);
    return true;
}

//-----------------------------------------------------------------------------
///
/// Helper that converts a python object to an int.
/// Returns true if the conversion succeeds.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::PyObject_to_int(PyObject *py_obj, int &res)
{
    if(PyInt_Check(py_obj))
    {
        res = (int)PyInt_AS_LONG(py_obj);
        return true;
    }

    if(PyLong_Check(py_obj))
    {
        res = (int)PyLong_AsLong(py_obj);
        return true;
    }

    if(PyNumber_Check(py_obj) != 1)
        return false;

    PyObject *py_val = PyNumber_Int(py_obj);
    if(py_val == NULL)
        return false;
    res = (int) PyInt_AS_LONG(py_val);
    Py_DECREF(py_val);
    return true;
}

//-----------------------------------------------------------------------------
///
/// Helper that converts a python object to a C++ string.
/// Returns true if the conversion succeeds.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::PyObject_to_string(PyObject *py_obj, std::string &res)
{
    PyObject *py_obj_str = PyObject_Str(py_obj);
    if(py_obj_str == NULL)
        return false;

    res = PyString_AS_STRING(py_obj_str);
    Py_DECREF(py_obj_str);
    return true;
}


//-----------------------------------------------------------------------------
///
/// Helper to turns a python traceback into a human readable string.
///
/// Note: Adapted from VisIt: src/avt/PythonFilters/PythonInterpreter.cpp
//-----------------------------------------------------------------------------
bool
PythonInterpreter::PyTraceback_to_string(PyObject *py_etype,
                                         PyObject *py_eval,
                                         PyObject *py_etrace,
                                         std::string &res)
{
    if(!py_eval)
        py_eval = Py_None;

    // create a StringIO object "buffer" to print traceback into.
    PyObject *py_args = Py_BuildValue("()");
    PyObject *py_buffer = PyObject_CallObject(m_py_sio_class, py_args);
    Py_DECREF(py_args);

    if(!py_buffer)
    {
        PyErr_Print();
        return false;
    }

    // call traceback.print_tb(etrace,file=buffer)
    PyObject *py_res = PyObject_CallFunction(m_py_trace_print_exception_func,
                                             (char*)"OOOOO",
                                             py_etype,
                                             py_eval,
                                             py_etrace,
                                             Py_None,
                                             py_buffer);
    if(!py_res)
    {
        PyErr_Print();
        return false;
    }

    // call buffer.getvalue() to get python string object
    PyObject *py_str = PyObject_CallMethod(py_buffer,(char*)"getvalue",NULL);
    

    if(!py_str)
    {
        PyErr_Print();
        return false;
    }

    // convert python string object to std::string
    res = PyString_AS_STRING(py_str);

    Py_DECREF(py_buffer);
    Py_DECREF(py_res);
    Py_DECREF(py_str);

    return true;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------




