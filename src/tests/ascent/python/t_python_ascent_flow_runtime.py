###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: t_python_ascent_flow.py
 description: Unit tests for basic ascent flow integration

"""


import sys
import unittest
import os

import conduit
import conduit.blueprint
import ascent
import flow


def inspect(val):
    print("Inspect sees" + str(val))
    return val

class Test_Ascent_Flow(unittest.TestCase):
    def test_render_flow_inspect(self):
        # if we don't have ascent, simply return
        info = ascent.about()
        if info["runtimes/ascent/status"] != "enabled":
            return

        flow.Workspace.register_builtin_filter_types()
        flow.Workspace.register_filter_type(flow.wrap_function(inspect));

        # create example mesh
        n_mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("quads",
                                              10,
                                              10,
                                              0,
                                              n_mesh)

        # open ascent
        a = ascent.Ascent()
        open_opts = conduit.Node()
        open_opts["runtime/type"].set("flow")
        a.open(open_opts)

        a.publish(n_mesh)

        actions = conduit.Node()
        add_py = actions.append()

        add_py["action"] = "add_filter";
        add_py["type_name"]  = "ensure_python";
        add_py["name"] = "py";

        add_ins = actions.append()
        add_ins["action"] = "add_filter";
        add_ins["type_name"]  = "inspect";
        add_ins["name"] = "my_inspect";


        conn_py = actions.append()
        conn_py["action"] = "connect";
        conn_py["src"]  = "source";
        conn_py["dest"] = "py";

        conn_ins = actions.append()
        conn_ins["action"] = "connect";
        conn_ins["src"]  = "py";
        conn_ins["dest"] = "my_inspect";

        print(actions)
        actions.append()["action"] = "execute"
        # this will fail for static libs case
        # b/c the py libs are still dynamic, and they each
        # link their own static copy of flow
        try:
            a.execute(actions)
        except RuntimeError as e:
            if not e.message.count("ensure_python") > 0:
                raise(e)
        a.close()

if __name__ == '__main__':
    unittest.main()



