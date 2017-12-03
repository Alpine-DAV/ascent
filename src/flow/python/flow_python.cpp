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

//-----------------------------------------------------------------------------
// we need the template specialization to be in the flow namespace
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// Template Specialization of Data Wrapper for PyObject *
//-----------------------------------------------------------------------------
    
template<>
class DataWrapper<PyObject>: public Data
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
            std::cout << "DataWrapper<PyObject> Decref" << std::endl;
            PyObject *py_obj =(PyObject*) data_ptr();
            Py_DECREF(py_obj);
            set_data_ptr(NULL);
        }
    }
};


//-----------------------------------------------------------------------------
// end namespace flow
//-----------------------------------------------------------------------------
};


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// Forward Decls
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
static PyObject *PyFlow_Graph_Python_Wrap(flow::Graph *graph);
static PyObject *PyFlow_Registry_Python_Wrap(Registry *registry);
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// Built in filter types that help with C++ to Python Conduit case
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
class PyFlow_Ensure_Python: public flow::Filter
{
public:

    //-------------------------------------------------------------------------
    PyFlow_Ensure_Python()
    : flow::Filter()
    {
        //empty
    }

    //-------------------------------------------------------------------------        
    virtual ~PyFlow_Ensure_Python()
    {
        //empty
    }
    
    //-------------------------------------------------------------------------
    virtual void declare_interface(conduit::Node &iface)
    {
        iface["type_name"]   = "ensure_python";
        iface["port_names"].append() = "in";
        iface["output_port"] = "true";
    }

    //-------------------------------------------------------------------------
    virtual void execute()
    {
        if(input(0).check_type<PyObject>())
        {
            // output is already python ... 
            set_output(input(0));
        }
        else if(input(0).check_type<conduit::Node>())
        {
            // wrap into python
            conduit::Node *n = input<conduit::Node>(0);
            PyObject *res = PyConduit_Node_Python_Wrap(n,
                                                       0);
            set_output<PyObject>(res);
        }
        else
        {
            CONDUIT_ERROR("ensure_python input must be a python object "
                          "or a conduit::Node");
        }
        
    }

};

//---------------------------------------------------------------------------//
class PyFlow_Ensure_Conduit: public flow::Filter
{
public:

    //-------------------------------------------------------------------------
    PyFlow_Ensure_Conduit()
    : flow::Filter()
    {
        //empty
    }
        
    virtual ~PyFlow_Ensure_Conduit()
    {
        //empty
    }

    //-------------------------------------------------------------------------    
    virtual void declare_interface(conduit::Node &iface)
    {
        iface["type_name"]   = "ensure_conduit";
        iface["port_names"].append() = "in";
        iface["output_port"] = "true";
    }

    //-------------------------------------------------------------------------
    virtual void execute()
    {
        if(input(0).check_type<conduit::Node>())
        {
            // output is already a Node ... 
            set_output(input(0));
        }
        else if(input(0).check_type<PyObject>())
        {
            PyObject *py_obj = input<PyObject>(0);
            
            if(PyConduit_Node_Check(py_obj))
            {
                CONDUIT_ERROR("ensure_conduit input must be a python wrapped "
                              "conduit::Node or a conduit::Node");
            }

            Node *node_ptr = PyConduit_Node_Get_Node_Ptr(py_obj);
            
            set_output<conduit::Node>(node_ptr);
        }
        else
        {
            CONDUIT_ERROR("ensure_conduit input must be a python wrapped "
                          "conduit::Node or a conduit::Node");
        }
        
    }

};

//---------------------------------------------------------------------------//
void
register_builtin_python_filter_types()
{

    if(!Workspace::supports_filter_type<PyFlow_Ensure_Python>())
    {
        Workspace::register_filter_type<PyFlow_Ensure_Python>();
    }

    if(!Workspace::supports_filter_type<PyFlow_Ensure_Conduit>())
    {
        Workspace::register_filter_type<PyFlow_Ensure_Conduit>();
    }

    
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// flow.Filter
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// base class for python filter integration
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
        // can we check for ref == 0 (already destructing)
        // ?
        // todo, when to decref m_py_obj ... ?
        // if(Py_REFCNT(m_py_obj) > 0)
        //    {
        //        Py_DECREF(m_py_obj);
        //    }

    }
    
    /// override and fill i with the info about the filter's interface
    virtual void declare_interface(conduit::Node &iface)
    {
        iface.reset();
        
        // call on m_py_obj
        PyObject *py_iface = PyConduit_Node_Python_Wrap(&iface,
                                                        0);

        PyObject *py_res = PyObject_CallMethod(m_py_obj,
                                               (char*)"declare_interface",
                                               (char*)"O",
                                                py_iface);
        if(py_res)
        {
            Py_DECREF(py_iface);
            Py_DECREF(py_res);
        }
        else
        {
            // TODO throw conduit exception?
            PyErr_Print();
        }
    }

    /// override to imp filter's work
    virtual void execute()
    {
        // call on m_py_obj
        PyObject *py_res = PyObject_CallMethod(m_py_obj,
                                               (char*)"execute",
                                               NULL);
        
        if(py_res)
        {
            Py_DECREF(py_res);
        }
        else
        {
            // TODO throw conduit exception?
            PyErr_Print();
        }
        
    }

    /// optionally override to allow filter to verify custom params
    /// (used as a guard when a filter instance is created in a graph)
    virtual bool verify_params(const conduit::Node &params,
                               conduit::Node &info)
    {
        // for methods using const, safest bet is to copy into
        // new python object
        PyObject *py_params = PyConduit_Node_Python_Create();
        PyConduit_Node_Get_Node_Ptr(py_params)->set(params);
        
        PyObject *py_info   = PyConduit_Node_Python_Wrap(&info,0);

        bool res = true;
        
        if(PyObject_HasAttrString(m_py_obj,"verify_params"))
        {

            // TODO CHECK AND CLEANUP RETURN 
            PyObject *py_res = PyObject_CallMethod(m_py_obj,
                                                   (char*)"verify_params",
                                                   (char*)"OO",
                                                   py_params,
                                                   py_info);

            if(py_res)
            {
                if(PyObject_IsTrue(py_res) == 1)
                {
                    res = true;
                }
                else
                {
                    res = false;
                }
                Py_DECREF(py_params);
                Py_DECREF(py_info);
                Py_DECREF(py_res);
            }
            else
            {
                // TODO throw conduit exception?
                PyErr_Print();
            }
        }
        
        return res;
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
static int PyFlow_Filter_Check(PyObject* obj);
//---------------------------------------------------------------------------//
static int PyFlow_Filter_Check_SubType(PyObject* obj);


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
    
    PyObject *res = PyConduit_Node_Python_Create();
    Node *node = PyConduit_Node_Get_Node_Ptr(res);
    node->set(self->filter->port_names());
    return res;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_default_params(PyFlow_Filter *self)
{
    
    PyObject *res = PyConduit_Node_Python_Create();
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
    
    PyObject *res = PyConduit_Node_Python_Create();
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
    PyObject *res = PyConduit_Node_Python_Wrap(&self->filter->params(),
                                               0);
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

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_output(PyFlow_Filter *self)
{
    return self->filter->output<PyObject>();
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Filter_graph(PyFlow_Filter *self)
{
    return PyFlow_Graph_Python_Wrap(&self->filter->graph());
}

//-----------------------------------------------------------------------------
PyObject *
PyFlow_Filter_connect_input_port(PyFlow_Filter *self, 
                                 PyObject *args,
                                 PyObject *kwargs)
{
    static const char *kwlist[] = {"port",
                                   "filter",
                                   NULL};
    
    const char *port_name_c_str = NULL;
    int port_idx = -1;
    PyObject *py_filter = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|O",
                                     const_cast<char**>(kwlist),
                                     &port_name_c_str))
    {
        if (!PyArg_ParseTupleAndKeywords(args,
                                         kwargs,
                                         "n|O",
                                         const_cast<char**>(kwlist),
                                         &port_idx))
        {
            PyErr_SetString(PyExc_TypeError,
                            "expects: "
                            "port(string), filter(flow.Filter)\n"
                            "or "
                            "port(index), filter(flow.Filter)\n");
            return NULL;
            
        }
    }
    
    if(!PyFlow_Filter_Check(py_filter))
    {
        PyErr_SetString(PyExc_TypeError,
                        "expects: "
                        "port(string), filter(flow.Filter)\n"
                        "or "
                        "port(index), filter(flow.Filter)\n");
        return NULL;
    }

    Filter *f = ((PyFlow_Filter*) py_filter)->filter;

    if(port_name_c_str != NULL)
    {
        self->filter->connect_input_port(std::string(port_name_c_str),f);
    }
    else
    {
        self->filter->connect_input_port(port_idx,f);
    }
    
    Py_RETURN_NONE;
}

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
                        "Filter.info 'info' argument must be a"
                        " conduit.Node");
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
PyFlow_Filter_str(PyFlow_Filter *self)
{
    return PyFlow_Filter_to_json(self);
}

//----------------------------------------------------------------------------//
// flow.Filter methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Filter_METHODS[] = {
    //-------------------------------------------------------------------------
    // filter interface properties
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"verify_interface",
      (PyCFunction)PyFlow_Filter_verify_interface,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      "checks if the given interface defintion is valid"},
    //-----------------------------------------------------------------------//
     {"type_name",
      (PyCFunction)PyFlow_Filter_type_name,
      METH_NOARGS,
      "returns this filters type name"},
    //-----------------------------------------------------------------------//
     {"port_names",
      (PyCFunction)PyFlow_Filter_port_names,
      METH_NOARGS,
      "returns names of this filters input ports"},
    //-----------------------------------------------------------------------//
     {"output_port",
      (PyCFunction)PyFlow_Filter_output_port,
      METH_NOARGS,
      "returns if this filter has an output port"},
    //-----------------------------------------------------------------------//
     {"default_params",
      (PyCFunction)PyFlow_Filter_default_params,
      METH_NOARGS,
      "returns a copy of the default parameters for this filter type"},
    //-----------------------------------------------------------------------//
     {"number_of_input_ports",
      (PyCFunction)PyFlow_Filter_number_of_input_ports,
      METH_NOARGS,
      "returns the number of input ports"},
    //-----------------------------------------------------------------------//
     {"has_port",
      (PyCFunction)PyFlow_Filter_has_port,
      METH_VARARGS,
      "checks if this filter has a port with the given name"},
    //-----------------------------------------------------------------------//
     {"port_index_to_name",
      (PyCFunction)PyFlow_Filter_port_index_to_name,
      METH_VARARGS,
      "returns the name of the port of the given index"},
    //-----------------------------------------------------------------------//
     {"name",
      (PyCFunction)PyFlow_Filter_name,
      METH_NOARGS,
      "returns this filters name"},
    //-----------------------------------------------------------------------//
     {"detailed_name",
      (PyCFunction)PyFlow_Filter_detailed_name,
      METH_NOARGS,
      "returns this filters detailed name"},
    //-------------------------------------------------------------------------
    // filter instance properties
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"interface",
     (PyCFunction)PyFlow_Filter_interface,
     METH_NOARGS ,
     "fills passed conduit.Node with this filter's interface definition"},
    //-----------------------------------------------------------------------//
     {"params",
     (PyCFunction)PyFlow_Filter_params,
     METH_NOARGS ,
     "fetches paramaters passed to this filter"},
    //-----------------------------------------------------------------------//
     {"input",
     (PyCFunction)PyFlow_Filter_input,
     METH_VARARGS ,
     "fetches a filter input"},
    //-----------------------------------------------------------------------//
     {"set_output",
     (PyCFunction)PyFlow_Filter_set_output,
     METH_VARARGS ,
     "sets the filter result"},
    //-----------------------------------------------------------------------//
     {"output",
     (PyCFunction)PyFlow_Filter_output,
     METH_NOARGS ,
     "gets filter output"},
    //-----------------------------------------------------------------------//
     {"graph",
     (PyCFunction)PyFlow_Filter_graph,
     METH_NOARGS ,
     "gets the graph that contains this filter"},
    //-----------------------------------------------------------------------//
     {"connect_input_port",
      (PyCFunction)PyFlow_Filter_connect_input_port,
      METH_VARARGS | METH_KEYWORDS,
      "connects the output of the passed filter to the given input port"},
    //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction)PyFlow_Filter_info,
      METH_VARARGS | METH_KEYWORDS,
      "fills passed conduit.Node with info about this filter"},
    //-----------------------------------------------------------------------//
    {"to_json",
     (PyCFunction)PyFlow_Filter_to_json,
     METH_NOARGS,
     "returns a json sting with info about this filter"},
    //-----------------------------------------------------------------------//
    // end flow.Filter methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyFlow_Filter_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "flow.Filter",
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
   "flow.Filter object",
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


//---------------------------------------------------------------------------//
static int
PyFlow_Filter_Check(PyObject *obj)
{
    return (PyObject_TypeCheck(obj, &PyFlow_Filter_TYPE));
}

//---------------------------------------------------------------------------//
static int
PyFlow_Filter_Check_SubType(PyObject *obj)
{
    if(!PyType_Check(obj))
    {
        return 0;
    }
    
    PyTypeObject *ptype = (PyTypeObject *)obj;
        
    return (PyType_IsSubtype(ptype, &PyFlow_Filter_TYPE));
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// flow.Registry
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
struct PyFlow_Registry
{
    PyObject_HEAD
    flow::Registry *registry;
};

//---------------------------------------------------------------------------//
static PyObject * 
PyFlow_Registry_new(PyTypeObject *type,
                    PyObject *, // args -- unused
                    PyObject *) // kwds -- unused
{
    PyFlow_Registry *self = (PyFlow_Registry*)type->tp_alloc(type, 0);

    if (self)
    {
        self->registry = 0;
    }

    return ((PyObject*)self);
}

//---------------------------------------------------------------------------//
static void
PyFlow_Registry_dealloc(PyFlow_Registry *self)
{
    // never owns a registry...
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyFlow_Registry_init(PyFlow_Registry *self,
                     PyObject *,// args -- unused
                     PyObject *) // kwds -- unused
{
  
    self->registry = 0;
    return 0;

}

//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_add(PyFlow_Registry *self, 
                      PyObject *args,
                      PyObject *kwargs)
{
    static const char *kwlist[] = {"key",
                                   NULL};
    
    const char *key_c_str  = NULL;
    PyObject *py_obj = NULL;
    Py_ssize_t num_refs = -1;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "sO|n",
                                     const_cast<char**>(kwlist),
                                     &key_c_str,
                                     &py_obj,
                                     &num_refs))
    {
        PyErr_SetString(PyExc_TypeError,
                        "expects: key(string), data(Python Object)| "
                        " refs_needed(integer)");
        return NULL;
    }

    self->registry->add<PyObject>(std::string(key_c_str),
                                  py_obj,
                                  num_refs);
    Py_RETURN_NONE;

}

//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_fetch(PyFlow_Registry *self, 
                      PyObject *args,
                      PyObject *kwargs)
{
    static const char *kwlist[] = {"key",
                                   NULL};
    
    const char *key_c_str  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &key_c_str))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'key' argument must be a string");
        return NULL;
    }

    return self->registry->fetch<PyObject>(std::string(key_c_str));
}


//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_has_entry(PyFlow_Registry *self, 
                          PyObject *args,
                          PyObject *kwargs)
{
    static const char *kwlist[] = {"key",
                                   NULL};
    
    const char *key_c_str  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &key_c_str))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'key' argument must be a string");
        return NULL;
    }

    if(self->registry->has_entry(std::string(key_c_str)))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_consume(PyFlow_Registry *self, 
                        PyObject *args,
                        PyObject *kwargs)
{
    static const char *kwlist[] = {"key",
                                   NULL};
    
    const char *key_c_str  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &key_c_str))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'key' argument must be a string");
        return NULL;
    }

    
    self->registry->consume(std::string(key_c_str));

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_detach(PyFlow_Registry *self, 
                       PyObject *args,
                       PyObject *kwargs)
{
    static const char *kwlist[] = {"key",
                                   NULL};
    
    const char *key_c_str  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s",
                                     const_cast<char**>(kwlist),
                                     &key_c_str))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'key' argument must be a string");
        return NULL;
    }

    
    self->registry->detach(std::string(key_c_str));

    Py_RETURN_NONE;
}


//-----------------------------------------------------------------------------
static PyObject *
PyFlow_Registry_info(PyFlow_Registry *self, 
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
                        "Filter.info 'info' argument must be a"
                        " conduit.Node");
        return NULL;
    }
    
    Node *info_ptr = PyConduit_Node_Get_Node_Ptr(py_info);
    
    self->registry->info(*info_ptr);

    Py_RETURN_NONE;

}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Registry_reset(PyFlow_Registry *self)
{
    self->registry->reset();
    Py_RETURN_NONE;
}



//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Registry_to_json(PyFlow_Registry *self)
{
    return Py_BuildValue("s", self->registry->to_json().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Registry_str(PyFlow_Registry *self)
{
    return PyFlow_Registry_to_json(self);
}

//----------------------------------------------------------------------------//
// flow.Registry methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Registry_METHODS[] = {
    {"add",
     (PyCFunction) PyFlow_Registry_add,
      METH_VARARGS | METH_KEYWORDS,
     "add entry to the registry"},
    {"fetch",
     (PyCFunction) PyFlow_Registry_fetch,
      METH_VARARGS | METH_KEYWORDS,
     "fetch an entry from the registry"},
    {"consume",
     (PyCFunction) PyFlow_Registry_consume,
      METH_VARARGS | METH_KEYWORDS,
     "consume an entry from the registry"},
    {"detach",
     (PyCFunction) PyFlow_Registry_detach,
      METH_VARARGS | METH_KEYWORDS,
     "detach an entry from the registry"},
    {"reset",
     (PyCFunction) PyFlow_Registry_reset,
     METH_NOARGS,
     "reset the registry"},
    //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction) PyFlow_Registry_info,
      METH_VARARGS | METH_KEYWORDS,
      "fills passed conduit.Node with info about this registry"},
    //-----------------------------------------------------------------------//
    {"to_json",
     (PyCFunction) PyFlow_Registry_to_json,
     METH_NOARGS,
     "returns a json sting with info about this registry"},
    //-----------------------------------------------------------------------//
    // end flow.Registry methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyFlow_Registry_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "flow.Registry",
   sizeof(PyFlow_Registry),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyFlow_Registry_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyFlow_Registry_str,                         /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "flow.Registry object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyFlow_Registry_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyFlow_Registry_init,
   0, /* alloc */
   PyFlow_Registry_new,   /* new */
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
static PyObject *
PyFlow_Registry_Python_Wrap(Registry *registry)
{
    PyTypeObject *type = (PyTypeObject*)&PyFlow_Registry_TYPE;
    PyFlow_Registry *res = (PyFlow_Registry*)type->tp_alloc(type, 0);

    res->registry = registry;
    return ((PyObject*)res);
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// flow.Graph
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
struct PyFlow_Graph
{
    PyObject_HEAD
    flow::Graph *graph;
};
//---------------------------------------------------------------------------//





//---------------------------------------------------------------------------//
static PyObject * 
PyFlow_Graph_new(PyTypeObject *type,
                 PyObject *, // args -- unused
                 PyObject *) // kwds -- unused
{
    PyFlow_Graph *self = (PyFlow_Graph*)type->tp_alloc(type, 0);

    if (self)
    {
        self->graph = 0;
    }

    return ((PyObject*)self);
}


//---------------------------------------------------------------------------//
static void
PyFlow_Graph_dealloc(PyFlow_Graph *self)
{
    // never owns a graph...
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyFlow_Graph_init(PyFlow_Graph *self,
                  PyObject *,// args -- unused
                  PyObject *) // kwds -- unused
{
    self->graph = 0;
    return 0;
}

/// TODO

    // //
    // //  /// access workspace that owns this graph
    // // Workspace &workspace();
    // //
    // /// check if this graph has a filter with passed name
    // bool has_filter(const std::string &name);
    //
    // /// remove if filter with passed name from this graph
    // void remove_filter(const std::string &name);
    // /// adds a set of filters and connections from the given graph
    // void add_graph(const Graph &g);
    // /// adds a set of filters and connections from a conduit tree that
    // //  describes them
    // void add_graph(const conduit::Node &g);
    //
    //
    // /// this methods are used by save() and info()
    // /// the produce conduit trees with data that can be used
    // /// add_filters() and add_connections().
    //
    // /// Provides a conduit description of the filters in the graph
    // void filters(conduit::Node &out) const;
    // /// Provides a conduit description of the connections in the graph
    // void connections(conduit::Node &out) const;
    //
    // /// adds a set of filters from a conduit tree that describes them
    // void add_filters(const conduit::Node &filters);
    // /// adds a set of connections from a conduit tree that describes them
    // void add_connections(const conduit::Node &conns);

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Graph_reset(PyFlow_Graph *self)
{
    self->graph->reset();
    
    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
PyObject *
PyFlow_Graph_add_filter(PyFlow_Graph *self,
                        PyObject *args,
                        PyObject *kwargs)
{
    static const char *kwlist[] = {"filter_type",
                                   "name",
                                   "params",
                                   NULL};

    const char *filter_type_c_str = NULL;
    const char *name_c_str = NULL;
    PyObject *py_params = NULL;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "s|sO",
                                     const_cast<char**>(kwlist),
                                     &filter_type_c_str,
                                     &name_c_str,
                                     &py_params))
    {
        PyErr_SetString(PyExc_TypeError,
                        "expects: "
                        "filter_type(string) | name(string), "
                        "params(conduit.Node)\n");
        return NULL;
    }


    std::string f_type(filter_type_c_str);
    
    Node *params_ptr = NULL;
    
    if(py_params != NULL)
    {
        if(!PyConduit_Node_Check(py_params))
        {
            PyErr_SetString(PyExc_TypeError,
                            "'params' must be a conduit.Node");
            return NULL;
        }

        params_ptr = PyConduit_Node_Get_Node_Ptr(py_params);
    }
    
    
    if(name_c_str == NULL)
    {
        if( py_params == NULL )
        {
            self->graph->add_filter(f_type);
        }
        else
        {
            self->graph->add_filter(f_type, *params_ptr);
        }
    }
    else
    {
        std::string f_name(name_c_str);

        if( params_ptr == NULL )
        {
            self->graph->add_filter(f_type, f_name);
        }
        else
        {   
            self->graph->add_filter(f_type, f_name, *params_ptr);
        }
    }

    Py_RETURN_NONE;

}


//-----------------------------------------------------------------------------
PyObject *
PyFlow_Graph_connect(PyFlow_Graph *self, 
                     PyObject *args,
                     PyObject *kwargs)
{
    static const char *kwlist[] = {"src",
                                   "dest",
                                   "port",
                                   NULL};

    const char *src_c_str  = NULL;
    const char *dest_c_str = NULL;

    const char *port_c_str = NULL;
    int port_idx = -1;
    
    // node case
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "ss|s",
                                     const_cast<char**>(kwlist),
                                     &src_c_str,
                                     &dest_c_str,
                                     &port_c_str))
    {
        if(!PyArg_ParseTupleAndKeywords(args,
                                        kwargs,
                                        "ss|n",
                                        const_cast<char**>(kwlist),
                                        &src_c_str,
                                        &dest_c_str,
                                        &port_idx))
        {
            PyErr_SetString(PyExc_TypeError,
                            "expects "
                            "src(string), dest(string), port(string)\n"
                            " or \n"
                            "src(string), dest(string), port(integer)\n");
            return NULL;
        }
    }
    
    std::string src(src_c_str);
    std::string dest(dest_c_str);

    // need either port name or index
    if( port_c_str != NULL)
    {
        self->graph->connect(src,dest,std::string(port_c_str));
    }
    else
    {
        self->graph->connect(src,dest,port_idx);
    }

    Py_RETURN_NONE;

}

//-----------------------------------------------------------------------------
PyObject *
PyFlow_Graph_save(PyFlow_Graph *self, 
                  PyObject *args,
                  PyObject *kwargs)
{
    static const char *kwlist[] = {"ofile",
                                   "protocol",
                                   "node",
                                   NULL};
    

    const char *ofile_c_str = NULL;
    const char *protocol_c_str = NULL;
    
    PyObject *py_node  = NULL;

    // node case
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|ssO",
                                     const_cast<char**>(kwlist),
                                     &ofile_c_str,
                                     &protocol_c_str,
                                     &py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "expects ofile(string), protocol(string) "
                        " or node(conduit.Node)");
        return NULL;
    }
    
    if(ofile_c_str!= NULL)
    {
        std::string ofile(ofile_c_str);
        if(protocol_c_str == NULL)
        {
            self->graph->save(ofile);
        }
        else
        {
            self->graph->save(ofile,std::string(protocol_c_str));
        }
    }
    else 
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "'node' must be a conduit.Node");
                            return NULL;
        }
        Node *node_ptr = PyConduit_Node_Get_Node_Ptr(py_node);
        self->graph->save(*node_ptr);
    }

    Py_RETURN_NONE;

}


//-----------------------------------------------------------------------------
PyObject *
PyFlow_Graph_load(PyFlow_Graph *self, 
                  PyObject *args,
                  PyObject *kwargs)
{
    static const char *kwlist[] = {"ofile",
                                   "protocol",
                                   "node",
                                   NULL};
    

    const char *ofile_c_str = NULL;
    const char *protocol_c_str = NULL;
    
    PyObject *py_node  = NULL;

    // node case
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "|ssO",
                                     const_cast<char**>(kwlist),
                                     &ofile_c_str,
                                     &protocol_c_str,
                                     &py_node))
    {
        PyErr_SetString(PyExc_TypeError,
                        "expects ofile(string), protocol(string) "
                        " or node(conduit.Node)");
        return NULL;
    }
    
    if(ofile_c_str!= NULL)
    {
        std::string ofile(ofile_c_str);
        if(protocol_c_str == NULL)
        {
            self->graph->load(ofile);
        }
        else
        {
            self->graph->load(ofile,std::string(protocol_c_str));
        }
    }
    else if(py_node != NULL)
    {
        if(!PyConduit_Node_Check(py_node))
        {
            PyErr_SetString(PyExc_TypeError,
                            "'node' must be a conduit.Node");
                            return NULL;
        }
        Node *node_ptr = PyConduit_Node_Get_Node_Ptr(py_node);
        self->graph->load(*node_ptr);
    }
    

    Py_RETURN_NONE;

}

//-----------------------------------------------------------------------------
PyObject *
PyFlow_Graph_info(PyFlow_Graph *self, 
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
                        "'info' argument must be a conduit.Node");
        return NULL;
    }
    
    Node *info_ptr = PyConduit_Node_Get_Node_Ptr(py_info);
    
    self->graph->info(*info_ptr);

    Py_RETURN_NONE;

}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Graph_to_json(PyFlow_Graph *self)
{
    return Py_BuildValue("s", self->graph->to_json().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Graph_to_dot(PyFlow_Graph *self)
{
    return Py_BuildValue("s", self->graph->to_dot().c_str());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Graph_str(PyFlow_Graph *self)
{
    return PyFlow_Graph_to_json(self);
}


//----------------------------------------------------------------------------//
// flow.Graph methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Graph_METHODS[] = {
    //-------------------------------------------------------------------------
    // instance methods
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"reset",
      (PyCFunction)PyFlow_Graph_reset,
      METH_NOARGS,
      "reset the graph"},
    //-----------------------------------------------------------------------//
     {"add_filter",
      (PyCFunction)PyFlow_Graph_add_filter,
      METH_VARARGS | METH_KEYWORDS,
      "add a filters to the graph"},
    //-----------------------------------------------------------------------//
     {"connect",
      (PyCFunction)PyFlow_Graph_connect,
      METH_VARARGS | METH_KEYWORDS,
      "connect filters in the graph"},
    //-----------------------------------------------------------------------//
     {"save",
      (PyCFunction)PyFlow_Graph_save,
      METH_VARARGS | METH_KEYWORDS,
      "save a graph to a file or a conduit.Node"},
    //-----------------------------------------------------------------------//
     {"load",
      (PyCFunction)PyFlow_Graph_load,
      METH_VARARGS | METH_KEYWORDS,
      "load a graph from a file or a conduit.Node"},
    //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction)PyFlow_Graph_info,
      METH_VARARGS | METH_KEYWORDS,
      "fills passed conduit.Node with info about this graph"},
    //-----------------------------------------------------------------------//
     {"to_json",
      (PyCFunction)PyFlow_Graph_to_json,
      METH_NOARGS,
      "returns a json sting with info about this graph"},
    //-----------------------------------------------------------------------//
     {"to_dot",
      (PyCFunction)PyFlow_Graph_to_json,
      METH_NOARGS,
      "returns a string with a grapviz dot style graph description"},
    //-----------------------------------------------------------------------//
    // end flow.Graph methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyFlow_Graph_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "flow.Graph",
   sizeof(PyFlow_Graph),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyFlow_Graph_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyFlow_Graph_str,   /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "flow.Graph object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyFlow_Graph_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyFlow_Graph_init,
   0, /* alloc */
   PyFlow_Graph_new,   /* new */
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
static PyObject *
PyFlow_Graph_Python_Wrap(Graph *graph)
{
    PyTypeObject *type = (PyTypeObject*)&PyFlow_Graph_TYPE;
    PyFlow_Graph *res = (PyFlow_Graph*)type->tp_alloc(type, 0);

    res->graph = graph;
    return ((PyObject*)res);
}


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//
// flow.Workspace
//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

class PyFlow_Workspace_Factory
{
public:
    static void register_filter_type(const std::string &filter_type_name,
                                     PyObject *py_cls)
    {
        // TODO GUARD
        m_map[filter_type_name] = py_cls;
    }    
    
    static PyFlow_Filter *create_py(const std::string &filter_type_name)
    {
        PyObject *py_cls = m_map[filter_type_name];
        PyObject *py_inst = PyObject_CallObject(py_cls, NULL);
        // TODO CHECK RET, etc
        return (PyFlow_Filter*)py_inst;
    }

    static flow::Filter *create(const std::string &filter_type_name)
    {
        PyFlow_Filter  *py_flow = create_py(filter_type_name);
        return py_flow->filter;
    }
    
    static void clear()
    {
        m_map.clear();
    }


private:
    static std::map<std::string,PyObject*> m_map;
    
};

std::map<std::string,PyObject*> PyFlow_Workspace_Factory::m_map;


//---------------------------------------------------------------------------//
struct PyFlow_Workspace
{
    PyObject_HEAD
    flow::Workspace *workspace;
};
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
static PyObject * 
PyFlow_Workspace_new(PyTypeObject *type,
                     PyObject *, // args -- unused
                     PyObject *) // kwds -- unused
{
    PyFlow_Workspace *self = (PyFlow_Workspace*)type->tp_alloc(type, 0);

    if (self)
    {
        self->workspace = 0;
    }

    return ((PyObject*)self);
}


//---------------------------------------------------------------------------//
static void
PyFlow_Workspace_dealloc(PyFlow_Workspace *self)
{
    if(self->workspace != NULL)
    {
        delete self->workspace;
    }
    
    Py_TYPE(self)->tp_free((PyObject*)self);
}


//---------------------------------------------------------------------------//
static int
PyFlow_Workspace_init(PyFlow_Workspace *self,
                      PyObject *,// args -- unused
                      PyObject *) // kwds -- unused
{
    self->workspace = new Workspace();
    return 0;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_graph(PyFlow_Workspace *self)
{
    Graph *g = &self->workspace->graph();
    return PyFlow_Graph_Python_Wrap(g);
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_registry(PyFlow_Workspace *self)
{
    Registry *g = &self->workspace->registry();
    return PyFlow_Registry_Python_Wrap(g);
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_traversals(PyFlow_Workspace *self,
                            PyObject *args,
                            PyObject *kwargs)
{
    static const char *kwlist[] = {"out",
                                   NULL};
    
    PyObject *py_out  = NULL;


    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "O",
                                     const_cast<char**>(kwlist),
                                     &py_out))
    {
        return (NULL);
    }

    if(!PyConduit_Node_Check(py_out))
    {
        PyErr_SetString(PyExc_TypeError,
                        "'out' argument must be a conduit.Node");
        return NULL;
    }
    
    Node *out_ptr = PyConduit_Node_Get_Node_Ptr(py_out);
    self->workspace->traversals(*out_ptr);
    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_execute(PyFlow_Workspace *self)
{
    self->workspace->execute();
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_reset(PyFlow_Workspace *self)
{
    self->workspace->reset();
    Py_RETURN_NONE;
}


//-----------------------------------------------------------------------------
PyObject *
PyFlow_Workspace_info(PyFlow_Workspace *self, 
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
                        "'info' argument must be a conduit.Node");
        return NULL;
    }
    
    Node *info_ptr = PyConduit_Node_Get_Node_Ptr(py_info);
    
    self->workspace->info(*info_ptr);

    Py_RETURN_NONE;

}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_to_json(PyFlow_Workspace *self)
{
    return Py_BuildValue("s", self->workspace->to_json().c_str());
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_str(PyFlow_Workspace *self)
{
    return PyFlow_Workspace_to_json(self);
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_set_default_mpi_comm(PyObject *, // cls -- unused
                                      PyObject *args)
{
    Py_ssize_t comm_id;
    if (!PyArg_ParseTuple(args, "n", &comm_id))
    {
        PyErr_SetString(PyExc_TypeError,
                "index must be an integer");
        return NULL;
    }
    
    Workspace::set_default_mpi_comm(comm_id);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_default_mpi_comm(PyObject *) // cls -- unused
{
    return PyLong_FromSsize_t((Py_ssize_t)Workspace::default_mpi_comm());
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_register_filter_type(PyObject *, // cls -- unused
                                      PyObject *args)
{
    // get python object
    PyObject *py_cls = NULL;
    if ( !PyArg_ParseTuple(args, "O", &py_cls) || 
         !PyFlow_Filter_Check_SubType(py_cls))
    {
        PyErr_SetString(PyExc_TypeError,
                        "type must be subclass of flow.Filter");
        return NULL;
    }

    // create instance
    PyObject *py_inst = PyObject_CallObject(py_cls, NULL);
    
    if(py_inst == NULL)
    {
        PyErr_Print();
    }
    
    // obtain pointer to filter
    flow::Filter *f = ((PyFlow_Filter*)py_inst)->filter;

    
    // TODO REMOVE:
    Node iface;
    f->declare_interface(iface);
        
    // this deletes f as well.
    Py_XDECREF(py_inst);
    
    std::string filter_type_name = iface["type_name"].as_string();

    PyFlow_Workspace_Factory::register_filter_type(filter_type_name,
                                                   py_cls);

    Workspace::register_filter_type(filter_type_name,
                                    &PyFlow_Workspace_Factory::create);
    
    Py_RETURN_NONE;
}


//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_supports_filter_type(PyObject *, // cls -- unused
                                      PyObject *args)
{
    const char *py_filter_type_name = NULL;
    if (!PyArg_ParseTuple(args, "s", &py_filter_type_name))
    {
        PyErr_SetString(PyExc_TypeError, "passed arg must be a string");
        return NULL;
    }

    if(Workspace::supports_filter_type(std::string(py_filter_type_name)))
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
PyFlow_Workspace_remove_filter_type(PyObject *, // cls -- unused
                                    PyObject *args)
{
    const char *py_filter_type_name = NULL;
    if (!PyArg_ParseTuple(args, "s", &py_filter_type_name))
    {
        PyErr_SetString(PyExc_TypeError,
                        "filter type name must be a string");
        return NULL;
    }

    Workspace::remove_filter_type(std::string(py_filter_type_name));

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------//
static PyObject *
PyFlow_Workspace_clear_supported_filter_types(PyObject *) // cls -- unused
{
    // TODO python specific map is only cleared here, not 
    // by other (say, c++ calls to Workspace::clear_supported_filter_types()
    // in a rare case this could cause confusion
    PyFlow_Workspace_Factory::clear();
    Workspace::clear_supported_filter_types();

    Py_RETURN_NONE;
}

// TODO, we may not want this in Workspace, it doens't match cpp case
static PyObject *
PyFlow_Workspace_register_builtin_filter_types(PyObject *) // cls -- unused
{
    flow::filters::register_builtin();
    register_builtin_python_filter_types();
    Py_RETURN_NONE;
}


//----------------------------------------------------------------------------//
// flow.Workspace methods table
//----------------------------------------------------------------------------//
static PyMethodDef PyFlow_Workspace_METHODS[] = {
    //-------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // static methods
    //-------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"set_default_mpi_comm",
      (PyCFunction)PyFlow_Workspace_set_default_mpi_comm,
      METH_VARARGS | METH_STATIC,
      "set the default mpi comm to an MPI_Comm_c2f integer"},
    //-----------------------------------------------------------------------//
     {"default_mpi_comm",
      (PyCFunction)PyFlow_Workspace_set_default_mpi_comm,
      METH_VARARGS | METH_STATIC,
      "gets the default mpi comm as an MPI_Comm_c2f integer"},
    // ------------------------------------------------------------------------
    /// filter factory interface
    // ------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"register_filter_type",
      (PyCFunction)PyFlow_Workspace_register_filter_type,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      "register a new type of filter"},
    //-----------------------------------------------------------------------//
     {"supports_filter_type",
      (PyCFunction)PyFlow_Workspace_supports_filter_type,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      "check if a filter type name is already registered"},
    //-----------------------------------------------------------------------//
     {"remove_filter_type",
      (PyCFunction)PyFlow_Workspace_remove_filter_type,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      "removes a filter type by name"},
    //-----------------------------------------------------------------------//
     {"clear_supported_filter_types",
      (PyCFunction)PyFlow_Workspace_clear_supported_filter_types,
      METH_NOARGS | METH_STATIC,
      "removes all registered filter types"},
    //-----------------------------------------------------------------------//
     {"register_builtin_filter_types",
      (PyCFunction)PyFlow_Workspace_register_builtin_filter_types,
      METH_NOARGS | METH_STATIC,
      "registers all builtin filter types, including those specific to python"},
    //-------------------------------------------------------------------------
    // instance methods
    //-------------------------------------------------------------------------
    //-----------------------------------------------------------------------//
     {"graph",
      (PyCFunction)PyFlow_Workspace_graph,
      METH_NOARGS,
      "returns a this workspace's graph"},
    //-----------------------------------------------------------------------//
     {"registry",
      (PyCFunction)PyFlow_Workspace_registry,
      METH_NOARGS,
      "returns a this workspace's registry"},
    //-----------------------------------------------------------------------//
     {"execute",
      (PyCFunction)PyFlow_Workspace_execute,
      METH_NOARGS,
      "executes this workspace's filter graph"},
    //-----------------------------------------------------------------------//
     {"traversals",
      (PyCFunction)PyFlow_Workspace_traversals,
      METH_VARARGS | METH_KEYWORDS,
      "fills passed conduit.Node with details about graph traversals"},
    //-----------------------------------------------------------------------//
     {"info",
      (PyCFunction)PyFlow_Workspace_info,
      METH_VARARGS | METH_KEYWORDS,
      "fills passed conduit.Node with info about this workspace"},
    //-----------------------------------------------------------------------//
     {"to_json",
      (PyCFunction)PyFlow_Workspace_to_json,
      METH_NOARGS,
      "returns a json sting with info about this workspace"},
    //------------------------------------------------
    //-----------------------------------------------------------------------//
    // end flow.Workspace methods table
    //-----------------------------------------------------------------------//
    {NULL, NULL, 0, NULL}
};

//---------------------------------------------------------------------------//
static PyTypeObject PyFlow_Workspace_TYPE = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "flow.Workspace",
   sizeof(PyFlow_Workspace),  /* tp_basicsize */
   0, /* tp_itemsize */
   (destructor)PyFlow_Workspace_dealloc,   /* tp_dealloc */
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
   (reprfunc)PyFlow_Workspace_str,   /* str */
   0, /* getattro */
   0, /* setattro */
   0, /* asbuffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* flags */
   "flow.Workspace object",
   0, /* traverse */
   0, /* clear */
   0, /* tp_richcompare */
   0, /* tp_weaklistoffset */
   0, /* iter */
   0, /* iternext */
   PyFlow_Workspace_METHODS, /* METHODS */
   0, /* MEMBERS */
   0, /* get/set */
   0, /* tp_base */
   0, /* dict */
   0, /* descr_get */
   0, /* gescr_set */
   0, /* dictoffset */
   (initproc)PyFlow_Workspace_init,
   0, /* alloc */
   PyFlow_Workspace_new,   /* new */
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
        NULL,
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
                                        NULL);
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

    if (PyType_Ready(&PyFlow_Registry_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    
    if (PyType_Ready(&PyFlow_Graph_TYPE) < 0)
    {
        PY_MODULE_INIT_RETURN_ERROR;
    }

    if (PyType_Ready(&PyFlow_Workspace_TYPE) < 0)
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

    //-----------------------------------------------------------------------//
    // add Registry Type
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyFlow_Registry_TYPE);
    PyModule_AddObject(py_module,
                       "Registry",
                       (PyObject*)&PyFlow_Registry_TYPE);

    //-----------------------------------------------------------------------//
    // add Graph Type
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyFlow_Graph_TYPE);
    PyModule_AddObject(py_module,
                       "Graph",
                       (PyObject*)&PyFlow_Graph_TYPE);

    //-----------------------------------------------------------------------//
    // add Workspace Type
    //-----------------------------------------------------------------------//
    
    Py_INCREF(&PyFlow_Workspace_TYPE);
    PyModule_AddObject(py_module,
                       "Workspace",
                       (PyObject*)&PyFlow_Workspace_TYPE);


#if defined(IS_PY3K)
    return py_module;
#endif

}

