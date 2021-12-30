###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: t_python_ascent_render.py
 description: Unit tests for basic ascent rendering via python

"""


import sys
import unittest
import os

import conduit
import conduit.blueprint
import ascent

import numpy as np

class Test_Ascent_Render(unittest.TestCase):
    def test_render_2d(self):
        # if we don't have ascent, simply return
        info = ascent.about()
        if info["runtimes/ascent/vtkm/status"] != "enabled":
            return

        obase = "tout_python_ascent_render_2d"
        ofile = obase + "100.png"
        # clean up old results if they exist
        if os.path.isfile(ofile):
            os.remove(ofile)

        # create example mesh
        n_mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("quads",
                                              10,
                                              10,
                                              0,
                                              n_mesh)

        # open ascent
        a = ascent.Ascent()
        a.open()

        a.publish(n_mesh)

        actions = conduit.Node()
        scenes  = conduit.Node()
        scenes["s1/plots/p1/type"] = "pseudocolor"
        scenes["s1/plots/p1/field"] = "braid"
        scenes["s1/image_prefix"] = obase

        add_act =actions.append()
        add_act["action"] = "add_scenes"
        add_act["scenes"] = scenes

        a.execute(actions)

        ascent_info = conduit.Node()
        a.info(ascent_info)
        print(ascent_info)

        a.close()
        self.assertTrue(os.path.isfile(ofile))

    def test_render_3d(self):
        # if we don't have ascent, simply return
        info = ascent.about()
        if info["runtimes/ascent/vtkm/status"] != "enabled":
            return

        obase = "tout_python_ascent_render_3d"
        ofile = obase + "100.png"
        # clean up old results if they exist
        if os.path.isfile(ofile):
            os.remove(ofile)


        # create example mesh
        n_mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("hexs",
                                              10,
                                              10,
                                              10,
                                              n_mesh)

        # open ascent
        a = ascent.Ascent()
        a.open()

        a.publish(n_mesh)

        actions = conduit.Node()
        scenes  = conduit.Node()
        scenes["s1/plots/p1/type"] = "pseudocolor"
        scenes["s1/plots/p1/field"] = "braid"
        scenes["s1/image_prefix"] = obase

        add_act =actions.append()
        add_act["action"] = "add_scenes"
        add_act["scenes"] = scenes

        a.execute(actions)
        a.close()
        self.assertTrue(os.path.isfile(ofile))

    def test_delete_scene(self):
        # exec again, but this time remove a scene
        # tests regression related to internal book keeping
        # with graph setup
        
        mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("hexs",
                                              5,
                                              5,
                                              5,
                                              mesh)
        a = ascent.Ascent()
        opts = conduit.Node()
        opts["exceptions"] = "forward"
        a.open(opts)
        a.publish(mesh);

        actions = conduit.Node()

        add_act = actions.append()
        add_act["action"] = "add_pipelines"
        pipelines = add_act["pipelines"]
        pipelines["pl1/f1/type"] = "contour"
        contour_params = pipelines["pl1/f1/params"]
        contour_params["field"] = "braid"
        iso_vals = np.array([0.2, 0.4],dtype=np.float32)
        contour_params["iso_values"].set(iso_vals)
        
        add_act2 = actions.append()
        add_act2["action"] = "add_scenes"
        scenes = add_act2["scenes"]
        scenes["s1/plots/p1/type"] = "pseudocolor"
        scenes["s1/plots/p1/pipeline"] = "pl1"
        scenes["s1/plots/p1/field"] = "braid"
        # set the output file name (ascent will add ".png")
        scenes["s1/image_name"] = "out_pipeline_ex1_contour"
        scenes["s2/plots/p1/type"] = "pseudocolor"
        scenes["s2/plots/p1/pipeline"] = "pl1"
        scenes["s2/plots/p1/field"] = "braid"
        # set the output file name (ascent will add ".png")
        scenes["s2/image_name"] = "out_pipeline_ex1_contour_blah"

        # print our full actions tree
        print(actions.to_yaml())
        # execute the actions
        a.execute(actions)

        # now exec w/o s1
        scenes.remove(path="s1")

        # print our full actions tree
        print(actions.to_yaml())
        # execute the actions
        a.execute(actions)


if __name__ == '__main__':
    unittest.main()



