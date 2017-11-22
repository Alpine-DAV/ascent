//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
// -- Python includes (these must be included first) -- 
//-----------------------------------------------------------------------------
#include <Python.h>
#include <structmember.h>
#include "bytesobject.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <vector>

//-----------------------------------------------------------------------------
// PyVarObject_TAIL is used at the end of each PyVarObject def
// to make sure we have the correct number of initializers across python
// versions.
//-----------------------------------------------------------------------------
#ifdef Py_TPFLAGS_HAVE_FINALIZE
#define PyVarObject_TAIL ,0
#else
#define PyVarObject_TAIL
#endif

//---------------------------------------------------------------------------//
// conduit includes
//---------------------------------------------------------------------------//
#include "conduit.hpp"
#include "flow.hpp"
#include "flow_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"

using namespace conduit;
using namespace flow;

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Begin Functions to help with Python 2/3 Compatibility.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#if defined(IS_PY3K)

//-----------------------------------------------------------------------------
int
PyString_Check(PyObject *o)
{
    return PyUnicode_Check(o);
}

//-----------------------------------------------------------------------------
char *
PyString_AsString(PyObject *py_obj)
{
    char *res = NULL;
    if(PyUnicode_Check(py_obj))
    {
        PyObject * temp_bytes = PyUnicode_AsEncodedString(py_obj,
                                                          "ASCII",
                                                          "strict"); // Owned reference
        if(temp_bytes != NULL)
        {
            res = strdup(PyBytes_AS_STRING(temp_bytes));
            Py_DECREF(temp_bytes);
        }
        else
        {
            // TODO: Error
        }
    }
    else if(PyBytes_Check(py_obj))
    {
        res = strdup(PyBytes_AS_STRING(py_obj));
    }
    else
    {
        // TODO: ERROR or auto convert?
    }
    
    return res;
}

//-----------------------------------------------------------------------------
PyObject *
PyString_FromString(const char *s)
{
    return PyUnicode_FromString(s);
}

//-----------------------------------------------------------------------------
void
PyString_AsString_Cleanup(char *bytes)
{
    free(bytes);
}


//-----------------------------------------------------------------------------
int
PyInt_Check(PyObject *o)
{
    return PyLong_Check(o);
}

//-----------------------------------------------------------------------------
long
PyInt_AsLong(PyObject *o)
{
    return PyLong_AsLong(o);
}

#else // python 2.6+

//-----------------------------------------------------------------------------
#define PyString_AsString_Cleanup(c) { /* noop */ }

#endif

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End Functions to help with Python 2/3 Compatibility.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------



    // ------------------------------------------------------------------------
    /// Interface to set and obtain the MPI communicator.
    ///
    /// We use an integer handle from MPI_Comm_c2f to avoid
    /// a header dependency of mpi just for the handle. 
    ///
    // ------------------------------------------------------------------------
    void static set_default_mpi_comm(int mpi_comm_id);
    int  static default_mpi_comm();

    // ------------------------------------------------------------------------
    /// filter factory interface
    // ------------------------------------------------------------------------

    /// register a new type 
    static void register_filter_type(FilterFactoryMethod fr);
    /// check if type with given name is registered
    static bool supports_filter_type(const std::string &filter_type);
    /// check if type with given factory is registered
    static bool supports_filter_type(FilterFactoryMethod fr);
    
    /// remove type with given name if registered
    static void remove_filter_type(const std::string &filter_type);
    /// remove all registered types
    static void clear_supported_filter_types();



//---------------------------------------------------------------------------//
class PyFlowFilter: public flow::Filter
{
public:

    PyFlowFilter(PyObject *py_obj)
    : flow::Filter(),
      m_py_obj(py_obj)
    {}
        
    virtual ~PyFlowFilter()
    {

    }
    
    /// override and fill i with the info about the filter's interface
    virtual void declare_interface(conduit::Node &i)
    {
        // call on m_py_obj
    }
        

    /// override to imp filter's work
    virtual void execute()
    {
        // call on m_py_obj
    }


    /// optionally override to allow filter to verify custom params
    /// (used as a guard when a filter instance is created in a graph)
    virtual bool verify_params(const conduit::Node &params,
                               conduit::Node &info)
    {
        // call on m_py_obj
        return false;
    }

private:
    PyObject *m_py_obj;

};




//---------------------------------------------------------------------------//
struct PyFlow_Filter
{
    PyObject_HEAD
    PyFlowFilter *filter;
};


//---------------------------------------------------------------------------//
static PyObject * 
PyFlow_Filter_new(PyTypeObject *type,
                  PyObject *, // args -- unused
                  PyObject *) // kwds -- unused
{
    PyFlow_Filter *self = (PyFlow_Filter*)type->tp_alloc(type, 0);

    if (self)
    {
        self->filter = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyFlow_Filter_dealloc(PyFlow_Filter *self)
{
    if(self->filter != NULL)
    {
        delete self->filter;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyFlow_Filter_init(PyFlow_Filter *self,
                   PyObject *,// args -- unused
                   PyObject *) // kwds -- unused
{
  
    self->filter = new PyFlowFilter((PyObject *)self);
    return 0;

}

//-----------------------------------------------------------------------------
// filter interface properties
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_type_name(PyFlow_Filter *self)
{
    return Py_BuildValue("s",self->filter->type_name().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_output_port(PyFlow_Filter *self)
{
    if(self->filter->output_port())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_port_names(PyFlow_Filter *self)
{
    
    PyObject *res = PyConduit_Node_python_create();
    Node *node = PyConduit_Node_Get_Node_Ptr(res);
    node->set(self->filter->port_names());
    return res;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_default_params(PyFlow_Filter *self)
{
    
    PyObject *res = PyConduit_Node_python_create();
    Node *node = PyConduit_Node_Get_Node_Ptr(res);
    node->set(self->filter->default_params());
    return res;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_number_of_input_ports(PyFlow_Filter *self)
{
    return PyLong_FromSsize_t(
                (Py_ssize_t)self->filter->number_of_input_ports());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_has_port(PyFlow_Filter *self,
                       PyObject *args)
{
    const char *port_name;
    if (!PyArg_ParseTuple(args, "s", &port_name))
    {
        PyErr_SetString(PyExc_TypeError, "Port name must be a string");
        return NULL;
    }

    
    if(self->filter->has_port(std::string(port_name)))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_port_index_to_name(PyFlow_Filter *self,
                                 PyObject *args)
{
    Py_ssize_t idx;

    if (!PyArg_ParseTuple(args, "n", &idx))
    {
        PyErr_SetString(PyExc_TypeError,
                "index must be a signed integer");
        return NULL;
    }

    return Py_BuildValue("s",self->filter->port_index_to_name(idx).c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_name(PyFlow_Filter *self)
{
    return Py_BuildValue("s", self->filter->name().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_detailed_name(PyFlow_Filter *self)
{
    return Py_BuildValue("s", self->filter->detailed_name().c_str());
}


//-----------------------------------------------------------------------------
PyObject *
PyFlow_Filter_verify_interface(PyObject *, // cls -- unused
                               PyObject *args,
                               PyObject *kwargs)
{
    static const char *kwlist[] = {"iface",
                                   "info",
                                   NULL};
    
    PyObject *py_iface = NULL;
    PyObject *py_info  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OO",
                                     const_cast<char**>(kwlist),
                                     &py_iface, &py_info))
    {
        return (NULL);
    }
        
    
    if(!PyConduit_Node_Check(py_iface))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Filter::verify_interface 'iface' argument must be a "
                        "Conduit::Node");
        return NULL;
    }

    
    if(!PyConduit_Node_Check(py_info))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Filter::verify_interface 'info' argument must be a "
                        "Conduit::Node");
        return NULL;
    }
    
    Node *iface_ptr = PyConduit_Node_Get_Node_Ptr(py_iface);
    Node *info_ptr  = PyConduit_Node_Get_Node_Ptr(py_info);
    
    if(flow::Filter::verify_interface(*iface_ptr,*info_ptr))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }

}


//----------------------------------------------------------------------------
// filter instance  properties
//----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_interface(PyFlow_Filter *self)
{
    
    PyObject *res = PyConduit_Node_python_create();
    Node *node = PyConduit_Node_Get_Node_Ptr(res);
    
    const flow::Filter *const_filt = (const flow::Filter*)self->filter;
    const Node &iface = const_filt->interface();
    node->set(iface);
    return res;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_params(PyFlow_Filter *self)
{
    // TODO: this copies for now, doesn't return ref
    PyObject *res = PyConduit_Node_python_create();
    Node *node = PyConduit_Node_Get_Node_Ptr(res);
    node->set(self->filter->params());
    return res;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_input(PyFlow_Filter *self,
                    PyObject *args)
{
    const char *port_name = NULL;
    Py_ssize_t idx;
    
    if(!PyArg_ParseTuple(args, "s", &port_name))
    {
        if(!PyArg_ParseTuple(args, "n", &idx))
        {
            PyErr_SetString(PyExc_TypeError,
                            "Port must be a string or index");
            return NULL;
        }
    }
    
    PyObject *res = NULL;
    if( port_name != NULL)
    {
        res = (PyObject*) self->filter->input<PyObject>(std::string(port_name));
    }
    else
    {
        res = (PyObject*) self->filter->input<PyObject>(idx);
    }
    
    return res;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_set_output(PyFlow_Filter *self,
                         PyObject *args)
{
    PyObject *py_data=NULL;
    if(!PyArg_ParseTuple(args, "O", &py_data))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Data must be a python object");
        return NULL;
    }
    
    self->filter->set_output<PyObject>(py_data);

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
// TODO:
//-----------------------------------------------------------------------------

// /// generic set of wrapped output data
// void                   set_output(Data &data);
//
// /// generic access to wrapped output data
// PyObject              *output();
// /// access the filter's graph
// PyObject              *graph();
//
// /// connect helper
// /// equiv to:
// ///   graph().connect(f->name(),this->name(),port_name);
// void                  connect_input_port(const std::string &port_name,
//                                          Filter *filter);
//
// /// connect helper
// /// equiv to:
// ///   graph().connect(f->name(),this->name(),idx);
// void                  connect_input_port(int idx,
//                                          Filter *filter);


//-----------------------------------------------------------------------------
PyObject *
PyFlow_Filter_info(PyFlow_Filter *self, 
                   PyObject *args,
                   PyObject *kwargs)
{
    static const char *kwlist[] = {"info",
                                   NULL};
    
    PyObject *py_info  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O",
                                     const_cast<char**>(kwlist),
                                     &py_info))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_info))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Filter::info 'info' argument must be a"
                        " Conduit::Node");
        return NULL;
    }
    
    Node *info_ptr = PyConduit_Node_Get_Node_Ptr(py_info);
    
    self->filter->info(*info_ptr);

    Py_RETURN_NONE;

}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_to_json(PyFlow_Filter *self)
{
    return Py_BuildValue("s", self->filter->to_json().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_print(PyFlow_Filter *self)
{
    self->filter->print();
    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_str(PyFlow_Filter *self)
{
    return PyFlow_Filter_to_json(self);
}

//----------------------------------------------------------------------------//
// flow::Filter methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Filter_METHODS[] = {
    //-------------------------------------------------------------------------
    // filter interface properties
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"verify_interface",
      (PyCFunction)PyFlow_Filter_verify_interface,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"type_name",
      (PyCFunction)PyFlow_Filter_type_name,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"port_names",
      (PyCFunction)PyFlow_Filter_port_names,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"output_port",
      (PyCFunction)PyFlow_Filter_output_port,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"default_params",
      (PyCFunction)PyFlow_Filter_default_params,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"number_of_input_ports",
      (PyCFunction)PyFlow_Filter_number_of_input_ports,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"has_port",
      (PyCFunction)PyFlow_Filter_has_port,
      METH_VARARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"port_index_to_name",
      (PyCFunction)PyFlow_Filter_port_index_to_name,
      METH_VARARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"name",
      (PyCFunction)PyFlow_Filter_name,
      METH_NOARGS,
      "{todo}"},
    //-----------------------------------------------------------------------//
     {"detailed_name",
      (PyCFunction)PyFlow_Filter_detailed_name,
      METH_NOARGS,
      "{todo}"},
    //-------------------------------------------------------------------------
    // filter instance properties
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"interface",
     (PyCFunction)PyFlow_Filter_interface,
     METH_NOARGS ,
     "{todo}"},
    //-----------------------------------------------------------------------//
     {"params",
     (PyCFunction)PyFlow_Filter_params,
     METH_NOARGS ,
     "{todo}"},
    //-----------------------------------------------------------------------//
     {"input",
     (PyCFunction)PyFlow_Filter_input,
     METH_VARARGS ,
     "{todo}"},
    //-----------------------------------------------------------------------//
     {"set_output",
     (PyCFunction)PyFlow_Filter_set_output,
     METH_VARARGS ,
     "{todo}"},
    //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction)PyFlow_Filter_info,
      METH_VARARGS | METH_KEYWORDS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"to_json",
     (PyCFunction)PyFlow_Filter_to_json,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"print",
     (PyCFunction)PyFlow_Filter_print,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // end flow::Filter methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyFlow_Filter_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "flow::Filter",
   sizeof(PyFlow_Filter),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyFlow_Filter_dealloc,   /* tp_dealloc */
   0, /* tp_print */
   0, /* tp_getattr */
   0, /* tp_setattr */
   0, /* tp_compare */
   0, /* tp_repr */
   0, /* tp_as_number */
   0, /* tp_as_sequence */
   0, /* as_mapping */
   0, /* hash */
   0, /* call */
   (reprfunc)PyFlow_Filter_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "flow::Filter object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyFlow_Filter_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyFlow_Filter_init,
   0, /* alloc */
   PyFlow_Filter_new,   /* new */
   0, /* tp_free */
   0, /* tp_is_gc */
   0, /* tp_bases */
   0, /* tp_mro */
   0, /* tp_cache */
   0, /* tp_subclasses */
   0,  /* tp_weaklist */
   0,
   0
   PyVarObject_TAIL
};

//
// //---------------------------------------------------------------------------//
// // ascent::about
// //---------------------------------------------------------------------------//
// static PyObject *
// PyAscent_about()
// {
//     //create and return a node with the result of about
//     PyObject *py_node_res = PyConduit_Node_python_create();
//     Node *node = PyConduit_Node_Get_Node_Ptr(py_node_res);
//     ascent::about(*node);
//     return (PyObject*)py_node_res;
// }
//
// //---------------------------------------------------------------------------//
// // Python Module Method Defs
// //---------------------------------------------------------------------------//
// static PyMethodDef ascent_python_funcs[] =
// {
//     //-----------------------------------------------------------------------//
//     {"about",
//      (PyCFunction)PyAscent_about,
//       METH_NOARGS,
//       NULL},
//     //-----------------------------------------------------------------------//
//     // end ascent methods table
//     //-----------------------------------------------------------------------//
//     {NULL, NULL, METH_VARARGS, NULL}
// };

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Module Init Code
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

struct module_state
{
    PyObject *error;
};

//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Extra Module Setup Logic for Python3
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
//---------------------------------------------------------------------------//
static int
flow_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
flow_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef flow_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "flow_python",
        NULL,
        sizeof(struct module_state),
        NULL, // flow_python_funcs,
        NULL,
        flow_python_traverse,
        flow_python_clear,
        NULL
};


#endif

//---------------------------------------------------------------------------//
// The module init function signature is different between py2 and py3
// This macro simplifies the process of returning when an init error occurs.
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
#define PY_MODULE_INIT_RETURN_ERROR return NULL
#else
#define PY_MODULE_INIT_RETURN_ERROR return
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// Main entry point
//---------------------------------------------------------------------------//
extern "C" 
//---------------------------------------------------------------------------//
#if defined(IS_PY3K)
PyObject *FLOW_PYTHON_API PyInit_flow_python(void)
#else
void FLOW_PYTHON_API initflow_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *py_module = PyModule_Create(&flow_python_module_def);
#else
    PyObject *py_module = Py_InitModule((char*)"flow_python",
                                        NULL);//flow_python_funcs);
#endif


    if(py_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(py_module);
    
    st->error = PyErr_NewException((char*)"flow_python.Error",
                                   NULL,
                                   NULL);
    if (st->error == NULL)
    {
        Py_DECREF(py_module);
        PY_MODULE_INIT_RETURN_ERROR;
    }

    // setup for conduit python c api
    if(import_conduit() < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    //-----------------------------------------------------------------------//
    // init our custom types
    //-----------------------------------------------------------------------//

    if (PyType_Ready(&PyFlow_Filter_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }
    //-----------------------------------------------------------------------//
    // add Filter Type
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyFlow_Filter_TYPE);
    PyModule_AddObject(py_module,
                       "Filter",
                       (PyObject*)&PyFlow_Filter_TYPE);


#if defined(IS_PY3K)
    return py_module;
#endif

}

