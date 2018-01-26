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
/// file: flow_python_interpreter.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_PYTHON_INTERPRETER_HPP
#define FLOW_PYTHON_INTERPRETER_HPP

#include <Python.h>

#include <flow_exports.h>
#include <string>
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
///
/// Simple C++ Embeddable Python Interpreter.
/// 
/// Note: This is based on VisIt's Embedded Python Interpreter implementation:
///          src/avt/PythonFilters/PythonInterpreter.{h,cpp}
//-----------------------------------------------------------------------------

class FLOW_API PythonInterpreter
{
public:
                 PythonInterpreter();
    virtual     ~PythonInterpreter();

    /// instance lifetime control
    bool         initialize(int argc=0, char **argv=NULL);
    
    bool         is_running() { return m_running; }
    void         reset();
    void         shutdown();

    /// helper to add a system path to access new modules
    bool         add_system_path(const std::string &path);

    /// script exec
    bool         run_script(const std::string &script);
    bool         run_script_file(const std::string &fname);

    /// access to global dict
    bool         set_global_object(PyObject *py_obj,
                                   const std::string &name);

    PyObject    *get_global_object(const std::string &name);

    PyObject    *global_dict() { return m_py_global_dict; }

    /// error checking 
    bool         check_error();
    void         clear_error();
    std::string  error_message() const { return m_error_msg; }

    /// helpers to obtain values from basic objects
    static bool  PyObject_to_double(PyObject *py_obj,
                                    double &res);

    static bool  PyObject_to_string(PyObject *py_obj,
                                    std::string &res);
    
    static bool  PyObject_to_int(PyObject *py_obj,
                                 int &res);

private:
    bool         PyTraceback_to_string(PyObject *py_etype,
                                       PyObject *py_eval,
                                       PyObject *py_etrace,
                                       std::string &res);

    bool         m_handled_init;
    bool         m_running;
    bool         m_error;
    std::string  m_error_msg;

    PyObject    *m_py_main_module;
    PyObject    *m_py_global_dict;

    PyObject    *m_py_trace_module;
    PyObject    *m_py_sio_module;
    PyObject    *m_py_trace_print_exception_func;
    PyObject    *m_py_sio_class;

};


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


