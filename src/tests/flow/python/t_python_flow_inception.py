###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: t_python_flow_inception.py
 description: Tests python calling python script via C++ interp.

"""

import sys
import unittest

import conduit
import flow


py_script = """
val = flow_input() * 2
print(val)
flow_set_output(val)
"""

def src():
    return 21

class Test_Flow_Python_Inception(unittest.TestCase):
    def test_inception(self):
        flow.Workspace.clear_supported_filter_types()
        flow.Workspace.register_builtin_filter_types()
        flow.Workspace.register_filter_type(flow.wrap_function(src))

        w = flow.Workspace()

        w.graph().add_filter("src","s");
        #
        py_params = conduit.Node()
        #py_params["interface/module"] = "mymod"
        py_params["source"] = py_script

        w.graph().add_filter("python_script","py",py_params);

        w.graph().connect("s","py","in");

        print(w)
        w.execute();

        print("Reg from Py")
        print(w.registry())

        v = w.registry().fetch("py")

        self.assertEqual(v,42)




if __name__ == '__main__':
    unittest.main()


