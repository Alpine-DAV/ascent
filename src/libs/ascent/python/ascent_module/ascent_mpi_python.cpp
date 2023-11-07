//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
#include "ascent.hpp"
#include "ascent_mpi_python_exports.h"

// conduit python module capi header
#include "conduit_python.hpp"

using namespace conduit;
using namespace ascent;

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


// Ascent class:
    // void   open(); // open with default options
    // void   open(conduit::Node &options);
    // void   publish(conduit::Node &data);
    // void   execute(conduit::Node &actions);
    // void   close();


//---------------------------------------------------------------------------//
struct PyAscent_MPI_Ascent
{
    PyObject_HEAD
    Ascent *ascent; // NoteIterator is light weight, we can deal with copies
};

//---------------------------------------------------------------------------//
// Helper that promotes ascent error to python error
//---------------------------------------------------------------------------//
static void
PyAscent_MPI_Ascent_Error_To_PyErr(const conduit::Error &e)
{
    std::ostringstream oss;
    oss << "Ascent Error: " << e.message();
    PyErr_SetString(PyExc_RuntimeError,
                    oss.str().c_str());
}

//---------------------------------------------------------------------------//
// Helper that promotes ascent error to python error
//---------------------------------------------------------------------------//
static void
PyAscent_MPI_Cpp_Error_To_PyErr(const char *msg)
{
    std::ostringstream oss;
    oss << "Ascent Error: " << msg;
    PyErr_SetString(PyExc_RuntimeError,
                    oss.str().c_str());
}

//---------------------------------------------------------------------------//
static PyObject *
PyAscent_MPI_Ascent_new(PyTypeObject *type,
                        PyObject*, // args -- unused
                        PyObject*) // kwds -- unused
{
    PyAscent_MPI_Ascent *self = (PyAscent_MPI_Ascent*)type->tp_alloc(type, 0);

    if (self)
    {
        self->ascent = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyAscent_MPI_Ascent_dealloc(PyAscent_MPI_Ascent *self)
{
    if(self->ascent != NULL)
    {
        delete self->ascent;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyAscent_MPI_Ascent_init(PyAscent_MPI_Ascent *self,
                         PyObject *,// args -- unused
                         PyObject *) // kwds -- unused
{

    self->ascent = new Ascent();
    return 0;

}

//-----------------------------------------------------------------------------
static PyObject *
PyAscent_MPI_Ascent_open(PyAscent_MPI_Ascent *self,
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
                            "Ascent::Open 'options' argument must be a "
                            "conduit::Node");
            return NULL;
        }
    }

    if(py_node != NULL)
    {
        Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
        self->ascent->open(*node);
    }
    else
    {
        self->ascent->open();
    }


    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject *
PyAscent_MPI_Ascent_publish(PyAscent_MPI_Ascent *self,
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
                        "Ascent::Publish 'data' argument must be a "
                        "conduit::Node");
        return NULL;
    }

    Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
    self->ascent->publish(*node);

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject *
PyAscent_MPI_Ascent_execute(PyAscent_MPI_Ascent *self,
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
                        "Ascent::Execute 'actions' argument must be a "
                        "conduit::Node");
        return NULL;
    }

    Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
    self->ascent->execute(*node);

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject *
PyAscent_MPI_Ascent_info(PyAscent_MPI_Ascent *self,
                         PyObject *args,
                         PyObject *kwargs)
{

    static const char *kwlist[] = {"out",
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
                        "Ascent::Info 'out' argument must be a "
                        "conduit::Node");
        return NULL;
    }

    Node *node = PyConduit_Node_Get_Node_Ptr(py_node);
    self->ascent->info(*node);

    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
static PyObject *
PyAscent_MPI_Ascent_close(PyAscent_MPI_Ascent *self)
{
    self->ascent->close();
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyAscent_MPI_Ascent_str(PyAscent_MPI_Ascent *self)
{
    return (Py_BuildValue("s", "{Ascent}"));
}

//----------------------------------------------------------------------------//
// Ascent methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyAscent_MPI_Ascent_METHODS[] = {
    //-----------------------------------------------------------------------//
    {"open",
     (PyCFunction)PyAscent_MPI_Ascent_open,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"publish",
     (PyCFunction)PyAscent_MPI_Ascent_publish,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"execute",
     (PyCFunction)PyAscent_MPI_Ascent_execute,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"info",
     (PyCFunction)PyAscent_MPI_Ascent_info,
     METH_VARARGS | METH_KEYWORDS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    {"close",
     (PyCFunction)PyAscent_MPI_Ascent_close,
     METH_NOARGS,
     "{todo}"},
    //-----------------------------------------------------------------------//
    // end Ascent methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyAscent_MPI_Ascent_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "Ascent",
   sizeof(PyAscent_MPI_Ascent),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyAscent_MPI_Ascent_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyAscent_MPI_Ascent_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "Ascent object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyAscent_MPI_Ascent_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyAscent_MPI_Ascent_init,
   0, /* alloc */
   PyAscent_MPI_Ascent_new,   /* new */
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
// ascent::about
//---------------------------------------------------------------------------//
static PyObject *
PyAscent_MPI_about()
{
    //create and return a node with the result of about
    PyObject *py_node_res = PyConduit_Node_Python_Create();
    Node *node = PyConduit_Node_Get_Node_Ptr(py_node_res);
    ascent::about(*node);
    return (PyObject*)py_node_res;
}

//---------------------------------------------------------------------------//
// ascent::execute_callback
//---------------------------------------------------------------------------//
static PyObject *
PyAscent_MPI_execute_callback(PyObject *self,
                              PyObject *args)
{
    char *callback_name;
    PyObject *py_params = NULL;
    PyObject *py_output = NULL;

    if (!PyArg_ParseTuple(args,
                          "sOO",
                          &callback_name,
                          &py_params,
                          &py_output))
    {
        return NULL;
    }

    try
    {
        if(py_params != NULL && py_output != NULL)
        {
            if(!PyConduit_Node_Check(py_params))
            {
                PyErr_SetString(PyExc_TypeError,
                                "Ascent::execute_callback 'params' argument must be a "
                                "conduit::Node");
                return NULL;
            }
            else if (!PyConduit_Node_Check(py_output))
            {
                PyErr_SetString(PyExc_TypeError,
                                "Ascent::execute_callback 'output' argument must be a "
                                "conduit::Node");
                return NULL;
            }
            std::string callback_name_string = callback_name;
            Node *params = PyConduit_Node_Get_Node_Ptr(py_params);
            Node *output = PyConduit_Node_Get_Node_Ptr(py_output);
            ascent::execute_callback(callback_name, *params, *output);
            Py_RETURN_NONE;
        }
    }
    catch(conduit::Error e)
    {
        PyAscent_MPI_Ascent_Error_To_PyErr(e);
        return NULL;
    }
    // also try to bottle other errors, to prevent python
    // from crashing due to uncaught exception
    catch(std::exception &e)
    {
        PyAscent_MPI_Cpp_Error_To_PyErr(e.what());
        return NULL;
    }
    catch(...)
    {
        PyAscent_MPI_Cpp_Error_To_PyErr("unknown cpp exception thrown");
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
// Python Module Method Defs
//---------------------------------------------------------------------------//
static PyMethodDef ascent_mpi_python_funcs[] =
{
    //-----------------------------------------------------------------------//
    {"about",
     (PyCFunction)PyAscent_MPI_about,
      METH_NOARGS,
      NULL},
    //-----------------------------------------------------------------------//
    {"execute_callback",
     (PyCFunction)PyAscent_MPI_execute_callback,
      METH_VARARGS,
      NULL},
    //-----------------------------------------------------------------------//
    // end ascent methods table
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
ascent_mpi_python_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static int
ascent_mpi_python_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

//---------------------------------------------------------------------------//
static struct PyModuleDef ascent_python_module_def =
{
        PyModuleDef_HEAD_INIT,
        "ascent_mpi_python",
        NULL,
        sizeof(struct module_state),
        ascent_mpi_python_funcs,
        NULL,
        ascent_mpi_python_traverse,
        ascent_mpi_python_clear,
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
PyObject *ASCENT_MPI_PYTHON_API PyInit_ascent_mpi_python(void)
#else
void ASCENT_MPI_PYTHON_API initascent_mpi_python(void)
#endif
//---------------------------------------------------------------------------//
{
    //-----------------------------------------------------------------------//
    // create our main module
    //-----------------------------------------------------------------------//

#if defined(IS_PY3K)
    PyObject *py_module = PyModule_Create(&ascent_python_module_def);
#else
    PyObject *py_module = Py_InitModule((char*)"ascent_mpi_python",
                                             ascent_mpi_python_funcs);
#endif


    if(py_module == NULL)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    struct module_state *st = GETSTATE(py_module);

    st->error = PyErr_NewException((char*)"ascent_mpi_python.Error",
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

    if (PyType_Ready(&PyAscent_MPI_Ascent_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }
    //-----------------------------------------------------------------------//
    // add DataType
    //-----------------------------------------------------------------------//

    Py_INCREF(&PyAscent_MPI_Ascent_TYPE);
    PyModule_AddObject(py_module,
                       "Ascent",
                       (PyObject*)&PyAscent_MPI_Ascent_TYPE);


#if defined(IS_PY3K)
    return py_module;
#endif

}

