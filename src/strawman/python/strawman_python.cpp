//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
#include "strawman.hpp"
#include "strawman_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"

using namespace conduit;
using namespace strawman;

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


// Strawman class:
    // void   Open(); // open with default options
    // void   Open(conduit::Node &options);
    // void   Publish(conduit::Node &data);
    // void   Execute(conduit::Node &actions);
    // void   Close();


//---------------------------------------------------------------------------//
struct PyStrawman_Strawman
{
    PyObject_HEAD
    Strawman *strawman; // NoteIterator is light weight, we can deal with copies
};


//---------------------------------------------------------------------------//
static PyObject * 
PyStrawman_Strawman_new(PyTypeObject *type,
                        PyObject*, // args -- unused
                        PyObject*) // kwds -- unused
{
    PyStrawman_Strawman *self = (PyStrawman_Strawman*)type->tp_alloc(type, 0);

    if (self)
    {
        self->strawman = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyStrawman_Strawman_dealloc(PyStrawman_Strawman *self)
{
    if(self->strawman != NULL)
    {
        delete self->strawman;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyStrawman_Strawman_init(PyStrawman_Strawman *self,
                         PyObject *,// args -- unused
                         PyObject *) // kwds -- unused
{
  
    self->strawman = new Strawman();
    return 0;

}

//-----------------------------------------------------------------------------
static PyObject *
PyStrawman_Strawman_open(PyStrawman_Strawman *self,
                         PyObject *args,
                         PyObject *kwargs)
{

    static const char *kwlist[] = {"options",
                                    NULL};

     PyObject *py_node = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|O",
                                     const_cast<char**>(kwlist),
                                     &py_node))
    {
        return NULL;
    }
    
    
    if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "Strawman::Open 'options' argument must be a "
                            "conduit::Node");
            return NULL;
        }
    }
    
    if(py_node != NULL)
    {
        Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
        self->strawman->Open(*node);
    }
    else
    {
        self->strawman->Open();
    }


    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyStrawman_Strawman_publish(PyStrawman_Strawman *self,
                            PyObject *args,
                            PyObject *kwargs)
{

    static const char *kwlist[] = {"data",
                                    NULL};

     PyObject *py_node = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O",
                                     const_cast<char**>(kwlist),
                                     &py_node))
    {
        return NULL;
    }
    
     
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Strawman::Publish 'data' argument must be a "
                        "conduit::Node");
        return NULL;
    }
    
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
    self->strawman->Publish(*node);

    Py_RETURN_NONE; 
}

//-----------------------------------------------------------------------------
static PyObject *
PyStrawman_Strawman_execute(PyStrawman_Strawman *self,
                            PyObject *args,
                            PyObject *kwargs)
{

    static const char *kwlist[] = {"actions",
                                    NULL};

     PyObject *py_node = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O",
                                     const_cast<char**>(kwlist),
                                     &py_node))
    {
        return NULL;
    }
    
     
    if(!PyConduit_Node_Check(py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "Strawman::Execute 'actions' argument must be a "
                        "conduit::Node");
        return NULL;
    }
    
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
    self->strawman->Execute(*node);

    Py_RETURN_NONE; 
}

//---------------------------------------------------------------------------//
static PyObject *
PyStrawman_Strawman_close(PyStrawman_Strawman *self)
{
    self->strawman->Close();
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyStrawman_Strawman_str(PyStrawman_Strawman *self)
{
    return (Py_BuildValue("s", "{STRAW,MAN!}"));
}

//----------------------------------------------------------------------------//
// Strawman methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyStrawman_Strawman_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"open",
     (PyCFunction)PyStrawman_Strawman_open,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"publish",
     (PyCFunction)PyStrawman_Strawman_publish,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
     //-----------------------------------------------------------------------//
     {"execute",
      (PyCFunction)PyStrawman_Strawman_execute,
     METH_VARARGS | METH_KEYWORDS,
      "{todo}"},
    //-----------------------------------------------------------------------//
    {"close",
     (PyCFunction)PyStrawman_Strawman_close, 
     METH_NOARGS,
     "{todo}"}, 
    //-----------------------------------------------------------------------//
    // end Strawman methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyStrawman_Strawman_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Strawman",
   sizeof(PyStrawman_Strawman),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyStrawman_Strawman_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyStrawman_Strawman_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Strawman object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyStrawman_Strawman_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyStrawman_Strawman_init,
   0, /* alloc */
   PyStrawman_Strawman_new,   /* new */
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


//---------------------------------------------------------------------------//
// strawman::about
//---------------------------------------------------------------------------//
static PyObject *
PyStrawman_about()
{
    //create and return a node with the result of about
    PyObject *py_node_res = PyConduit_Node_python_create();
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node_res);
    strawman::about(*node);
    return (PyObject*)py_node_res;
}

//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef strawman_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    {"about",
     (PyCFunction)PyStrawman_about,
      METH_NOARGS,
      NULL},
    //-----------------------------------------------------------------------//
    // end strawman methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, METH_VARARGS, NULL}
};

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
strawman_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int 
strawman_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef relay_python_module_def = 
{
        PyModuleDef_HEAD_INIT,
        "strawman_python",
        NULL,
        sizeof(struct module_state),
        strawman_python_funcs,
        NULL,
        strawman_python_traverse,
        strawman_python_clear,
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
PyObject *STRAWMAN_PYTHON_API PyInit_strawman_python(void)
#else
void STRAWMAN_PYTHON_API initstrawman_python(void)
#endif
//---------------------------------------------------------------------------//
{    
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *py_module = PyModule_Create(&strawmanf_python_module_def);
#else
    PyObject *py_module = Py_InitModule((char*)"strawman_python",
                                             strawman_python_funcs);
#endif


    if(py_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(py_module);
    
    st->error = PyErr_NewException((char*)"strawman_python.Error",
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

    if (PyType_Ready(&PyStrawman_Strawman_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }
    //-----------------------------------------------------------------------//
    // add DataType
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyStrawman_Strawman_TYPE);
    PyModule_AddObject(py_module,
                       "Strawman",
                       (PyObject*)&PyStrawman_Strawman_TYPE);


#if defined(IS_PY3K)
    return py_module;
#endif

}

