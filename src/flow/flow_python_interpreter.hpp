//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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

    /// Note: blows away everything in the main dict
    /// use with caution!
    void         reset();
    void         shutdown();

    /// echo (default = false)
    ///  when enabled, controls if contents of execd python 
    //   scripts are echoed to conduit info
    bool         echo_enabled() const { return m_echo; }
    /// change echo setting
    void         set_echo(bool value) { m_echo = value; }

    void         set_program_name(const char *name);
    void         set_argv(int argc, char **argv);

    /// helper to add a system path to access new modules
    bool         add_system_path(const std::string &path);

    /// script exec
    bool         run_script(const std::string &script);
    bool         run_script_file(const std::string &fname);
    
    /// script exec in specific dict
    bool         run_script(const std::string &script,
                            PyObject *py_dict);
    bool         run_script_file(const std::string &fname,
                                 PyObject *py_dict);

    /// set into global dict
    bool         set_global_object(PyObject *py_obj,
                                   const std::string &name);
    /// fetch from global dict, returns borrowed reference
    PyObject    *get_global_object(const std::string &name);
    /// access global dict object
    PyObject    *global_dict() { return m_py_global_dict; }

    /// set into given dict
    bool         set_dict_object(PyObject *py_dict,
                                 PyObject *py_obj,
                                 const std::string &name);
    /// fetch from given dict, returns borrowed reference
    PyObject    *get_dict_object(PyObject *py_dict, 
                                 const std::string &name);

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
    bool         m_echo;
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


