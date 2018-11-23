###############################################################################
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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

        actions.append()["action"] = "execute"

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

        actions.append()["action"] = "execute"

        a.execute(actions)
        a.close()
        self.assertTrue(os.path.isfile(ofile))


if __name__ == '__main__':
    unittest.main()



