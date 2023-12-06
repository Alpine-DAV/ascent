###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: helpers.py
# Purpose: TODO
###############################################################################

from .flow_python import Filter
import inspect

def wrap_function(func):
    """
    Creates a filter class from a plain python function
    """
    class FilterWrap(Filter):
        def __init__(self):
             self.func = func
             super(FilterWrap, self).__init__()
        def declare_interface(self,i):
            i.fetch("type_name").set(self.func.__name__);
            i.fetch("output_port").set("true");
            sig = inspect.signature(self.func)
            for k in sig.parameters:
                i["port_names"].append().set(k)
        def execute(self):
            arg_vals = []
            sig = inspect.signature(self.func)
            for k in sig.parameters:
                arg_vals.append(self.input(k))
            self.set_output(func(*arg_vals))
    return FilterWrap
