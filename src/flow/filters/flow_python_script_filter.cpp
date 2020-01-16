//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
/// file: flow_python_script_filter.cpp
///
//-----------------------------------------------------------------------------

// always include python's headers first
#include <Python.h>

#include "flow_python_script_filter.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

// conduit python module capi header
#include "conduit_python.hpp"
//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_workspace.hpp>
#include <flow_python_interpreter.hpp>

using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
/// The FLOW_CHECK_PYTHON_ERROR macro is used to check errors from Python.
/// Must be called inside of PythonScript class.
//-----------------------------------------------------------------------------

#define FLOW_CHECK_PYTHON_ERROR( py_interp, py_ok )                 \
{                                                                   \
    if(  !py_ok  )                                                  \
    {                                                               \
        CONDUIT_ERROR("python interpreter failure:" <<              \
                      py_interp->error_message());                  \
    }                                                               \
}


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// Make sure we treat cleanup of python objects correctly
//-----------------------------------------------------------------------------
template<>
class FLOW_API DataWrapper<PyObject>: public Data
{
 public:
    DataWrapper(void *data)
    : Data(data)
    {
        // empty
    }

    virtual ~DataWrapper()
    {
        // empty
    }

    Data *wrap(void *data)
    {
        return new DataWrapper<PyObject>(data);
    }

    virtual void release()
    {
        if(data_ptr() != NULL)
        {
            PyObject *py_obj =(PyObject*) data_ptr();
            Py_DECREF(py_obj);
            set_data_ptr(NULL);
        }
    }
};


//-----------------------------------------------------------------------------
// -- begin flow::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin flow::filters::detail --
//-----------------------------------------------------------------------------
namespace detail
{

PyObject* execute_python(PyObject *py_input,
                        flow::PythonInterpreter *py_interp,
                        conduit::Node &params)
{
    std::string module_name = "flow_script_filter";
    std::string input_func_name = "flow_input";
    std::string set_output_func_name = "flow_set_output";

    bool echo = false;
    if( params.has_path("echo") &&
        params["echo"].as_string() == "true")
    {
        echo = true;
    }

    py_interp->set_echo(echo);

    if( params.has_path("interface/module") )
    {
        module_name = params["interface/module"].as_string();
    }

    if( params.has_path("interface/input") )
    {
        input_func_name = params["interface/input"].as_string();
    }

    if( params.has_path("interface/set_output") )
    {
        set_output_func_name = params["interface/set_output"].as_string();
    }

    std::ostringstream filter_setup_src_oss;
    // lookup or create a new module
    filter_setup_src_oss.str("");
    filter_setup_src_oss << "def flow_setup_module(name):\n"
                         << "    import sys\n"
                         << "    import imp\n"
                         << "    if name in sys.modules.keys():\n"
                         << "       return sys.modules[name]\n"
                         << "    mymod = imp.new_module(name)\n"
                         << "    sys.modules[name] = mymod\n"
                         << "    return mymod\n"
                         << "\n"
                         // setup the module
                         << "flow_setup_module(\"" << module_name << "\")\n";
    FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script(filter_setup_src_oss.str()));

    filter_setup_src_oss.str();
    filter_setup_src_oss << "\n"
                         // import into the global dict
                         << "import " << module_name << "\n"
                         << "\n";
    FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script(filter_setup_src_oss.str()));


    // fetch the module from the global dict (borrowed)
    PyObject *py_mod = py_interp->get_global_object(module_name);

    // sanity check
    if( !PyModule_Check(py_mod) )
    {
        CONDUIT_ERROR("Unexpected error: " << module_name
                      << " is not a python module!");
    }

    // then grab the module's dict (borrowed)
    //  where we will place our methods and bind our input data
    PyObject *py_mod_dict = PyModule_GetDict(py_mod);

    // bind our input data
    FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->set_dict_object(py_mod_dict,
                                                                  py_input,
                                                                  "_flow_input"));

    // run script to establish input and output helpers in the module
    // note: global here binds to module scope
    filter_setup_src_oss.str("");
    filter_setup_src_oss << "\n"
                         << "_flow_output = None\n"
                         << "\n"
                         << "def "<< input_func_name << "():\n"
                         << "    return _flow_input\n"
                         << "\n"
                         << "def " << set_output_func_name <<  "(out):\n"
                         << "    global _flow_output\n"
                         << "    _flow_output = out\n"
                         << "\n";

    FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script(filter_setup_src_oss.str(),
                                                             py_mod_dict));

    // now import binding function names from the module,
    // so the names are bound to the global ns
    filter_setup_src_oss.str("");
    filter_setup_src_oss << "\n"
                         << "from " << module_name
                         << " import "
                         << input_func_name << ", "
                         << set_output_func_name
                         << "\n";

    FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script(filter_setup_src_oss.str()));


    if( params.has_child("source") )
    {
        FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script(params["source"].as_string()));
    }
    else // file is the other case
    {
        FLOW_CHECK_PYTHON_ERROR(py_interp, py_interp->run_script_file(params["file"].as_string()));
    }

    PyObject *py_res = py_interp->get_dict_object(py_mod_dict,
                                                  "_flow_output");

    if(py_res == NULL)
    {
        // bad!, it should at least be python's None
        CONDUIT_ERROR("python_script failed to fetch output");
    }

    // we need to incref b/c py_res is borrowed, and flow will decref
    // when it is done with the python object
    Py_INCREF(py_res);
    return py_res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow::filters::detail --
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
flow::PythonInterpreter *PythonScript::m_interp = NULL;

flow::PythonInterpreter *PythonScript::interpreter()
{
    if(m_interp == NULL)
    {
        m_interp = new PythonInterpreter();

        if(!m_interp->initialize())
        {
            delete m_interp;
            m_interp = NULL;
            CONDUIT_ERROR("PythonInterpreter initialize failed");
        }

        // setup for conduit python c api
        if(!m_interp->run_script("import conduit"))
        {
            std::string emsg = interpreter()->error_message();
            delete m_interp;
            m_interp = NULL;
            CONDUIT_ERROR("failed to import conduit\n"
                           << emsg);
        }

        if(import_conduit() < 0)
        {
            delete m_interp;
            m_interp = NULL;
            CONDUIT_ERROR("failed to import Conduit Python C-API");
        }
    }

    return m_interp;
}

//-----------------------------------------------------------------------------
PythonScript::PythonScript()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
PythonScript::~PythonScript()
{
// empty
}

//-----------------------------------------------------------------------------
void
PythonScript::declare_interface(Node &i)
{
    i["type_name"] = "python_script";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
PythonScript::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = true;

    if( params.has_child("source") )
    {
        if( !params["source"].dtype().is_string() )
        {
            info["errors"].append() = "parameter 'source' is not a string";
            res = false;
        }
    }
    else if( params.has_child("file") )
    {
        if( !params["file"].dtype().is_string() )
        {
            info["errors"].append() = "parameter 'file' is not a string";
            res = false;
        }
    }
    else
    {
        info["errors"].append() = "Missing required string parameter"
                                  " 'source' or 'file'";
        res = false;
    }

    if( params.has_child("echo") )
    {
        if( !params["echo"].dtype().is_string() ||
             (params["echo"].as_string() != "true" &&
              params["echo"].as_string() != "false") )
        {
            info["errors"].append() = "parameter 'echo' is not \"true\" or \"false\"";
            res = false;
        }
    }

    if( params.has_child("interface") )
    {
        const Node &n_iface = params["interface"];

        if( n_iface.has_child("module") )
        {
            if( !n_iface["module"].dtype().is_string() )
            {
                info["errors"].append() = "parameter 'interface/module' is not a string";
                res = false;
            }
            else
            {
                info["info"].append().set("provides 'interface/module' module name override");
            }
        }

        if( n_iface.has_child("input") )
        {
            if( !n_iface["input"].dtype().is_string() )
            {
                info["errors"].append() = "parameter 'interface/input' is not a string";
                res = false;
            }
            else
            {
                info["info"].append().set("provides 'interface/input' function name override");
            }
        }

        if( n_iface.has_child("set_output") )
        {
            if( !n_iface["set_output"].dtype().is_string() )
            {
                info["errors"].append() = "parameter 'interface/set_output' is not a string";
                res = false;
            }
            else
            {
                info["info"].append().set("provides 'interface/set_output' function name override");
            }
        }

    }

    return res;
}

//-----------------------------------------------------------------------------
void PythonScript::execute_python(conduit::Node *n)
{
    PyObject * py_input = PyConduit_Node_Python_Wrap(n,0);
    PythonInterpreter *py_interp = interpreter();

    PyObject *py_res = detail::execute_python(py_input, py_interp, params());
    set_output<PyObject>(py_res);
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
PythonScript::execute()
{
    // make sure we have our interpreter setup b/c
    // we need the python env ready

    if(input(0).check_type<PyObject>())
    {
        // input is already have python
        PyObject *py_input = NULL;
        py_input = input<PyObject>(0);
        PythonInterpreter *py_interp = interpreter();
        PyObject *py_res = detail::execute_python(py_input, py_interp, params());
        set_output<PyObject>(py_res);
    }
    else if(input(0).check_type<conduit::Node>())
    {
        // input is conduit node, wrap into python
        conduit::Node *n = input<conduit::Node>(0);
        execute_python(n);
    }
    else
    {
        CONDUIT_ERROR("python_script input must be a python object "
                      "or a conduit::Node");
    }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow::filters --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------



