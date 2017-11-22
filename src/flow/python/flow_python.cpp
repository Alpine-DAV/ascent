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


//
//
// //-----------------------------------------------------------------------------
// static PyObject *
// PyFlow_Filter_open(PyFlow_Filter *self,
//                    PyObject *args,
//                    PyObject *kwargs)
// {
//
//     static const char *kwlist[] = {"iface",
//                                     NULL};
//
//      PyObject *py_node = NULL;
//
//     if (!PyArg_ParseTupleAndKeywords(args,
//                                      kwargs,
//                                      "O",
//                                      const_cast<char**>(kwlist),
//                                      &py_node))
//     {
//         return NULL;
//     }
//
//
//     if(py_node != NULL)
//     {
//         if(!PyConduit_Node_Check(py_node))
//         {
//             PyErr_SetString(PyExc_TypeError,
//                             "Ascent::Open 'options' argument must be a "
//                             "conduit::Node");
//             return NULL;
//         }
//     }
//
//     if(py_node != NULL)
//     {
//         Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
//         self->ascent->open(*node);
//     }
//     else
//     {
//         self->ascent->open();
//     }
//
//
//     Py_RETURN_NONE;
// }
//
// //-----------------------------------------------------------------------------
// static PyObject *
// PyAscent_Ascent_publish(PyAscent_Ascent *self,
//                             PyObject *args,
//                             PyObject *kwargs)
// {
//
//     static const char *kwlist[] = {"data",
//                                     NULL};
//
//      PyObject *py_node = NULL;
//
//     if (!PyArg_ParseTupleAndKeywords(args,
//                                      kwargs,
//                                      "O",
//                                      const_cast<char**>(kwlist),
//                                      &py_node))
//     {
//         return NULL;
//     }
//
//
//     if(!PyConduit_Node_Check(py_node))
//     {
//         PyErr_SetString(PyExc_TypeError,
//                         "Ascent::Publish 'data' argument must be a "
//                         "conduit::Node");
//         return NULL;
//     }
//
//     Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
//     self->ascent->publish(*node);
//
//     Py_RETURN_NONE;
// }
//
// //-----------------------------------------------------------------------------
// static PyObject *
// PyAscent_Ascent_execute(PyAscent_Ascent *self,
//                             PyObject *args,
//                             PyObject *kwargs)
// {
//
//     static const char *kwlist[] = {"actions",
//                                     NULL};
//
//      PyObject *py_node = NULL;
//
//     if (!PyArg_ParseTupleAndKeywords(args,
//                                      kwargs,
//                                      "O",
//                                      const_cast<char**>(kwlist),
//                                      &py_node))
//     {
//         return NULL;
//     }
//
//
//     if(!PyConduit_Node_Check(py_node))
//     {
//         PyErr_SetString(PyExc_TypeError,
//                         "Ascent::Execute 'actions' argument must be a "
//                         "conduit::Node");
//         return NULL;
//     }
//
//     Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
//     self->ascent->execute(*node);
//
//     Py_RETURN_NONE;
// }
//
// //---------------------------------------------------------------------------//
// static PyObject *
// PyAscent_Ascent_close(PyAscent_Ascent *self)
// {
//     self->ascent->close();
//     Py_RETURN_NONE;
// }


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
    // TODO: use to_json()
    return (Py_BuildValue("s", "{Python Flow Filter}"));
}

//----------------------------------------------------------------------------//
// Ascent methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Filter_METHODS[] = {
    //-----------------------------------------------------------------------//
    // {"open",
    //  (PyCFunction)PyAscent_Ascent_open,
    //  METH_VARARGS | METH_KEYWORDS,
    //  "{todo}"},
    // //-----------------------------------------------------------------------//
    // {"publish",
    //  (PyCFunction)PyAscent_Ascent_publish,
    //  METH_VARARGS | METH_KEYWORDS,
    //  "{todo}"},
    //  //-----------------------------------------------------------------------//
    //  {"execute",
    //   (PyCFunction)PyAscent_Ascent_execute,
    //  METH_VARARGS | METH_KEYWORDS,
    //   "{todo}"},
    //-----------------------------------------------------------------------//
    {"print",
     (PyCFunction)PyFlow_Filter_print,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // end Ascent methods table
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

