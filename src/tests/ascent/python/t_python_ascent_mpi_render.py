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
import ascent.mpi

from mpi4py import MPI


class Test_Ascent_MPI_Render(unittest.TestCase):
    def test_render_mpi_2d(self):
        # if we don't have ascent, or mpi simply return

        info = ascent.mpi.about()
        if info["runtimes/ascent/vtkm/status"] != "enabled":
            print("ascent runtime not enabled, skipping mpi render test")
            return


        obase = "tout_python_ascent_mpi_render_2d"
        ofile = obase + "100.png"
        # clean up old results if they exist
        if MPI.COMM_WORLD.rank == 0 and os.path.isfile(ofile):
            os.remove(ofile)

        # create example mesh using conduit blueprint
        n_mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("quads",
                                              10,
                                              10,
                                              0,
                                              n_mesh)
        # shift data for rank > 1
        x_origin = MPI.COMM_WORLD.rank * 20 - 10;

        n_mesh["state/domain_id"] = MPI.COMM_WORLD.rank
        n_mesh["coordsets/coords/origin/x"] = x_origin
        n_mesh["fields/braid/values"][:] = MPI.COMM_WORLD.rank


        # open ascent
        a = ascent.mpi.Ascent()
        ascent_opts = conduit.Node()
        ascent_opts["mpi_comm"].set(MPI.COMM_WORLD.py2f())
        a.open(ascent_opts)

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

        # use barrier to avoid problems
        # some ranks may finish before file is written
        MPI.COMM_WORLD.Barrier()
        self.assertTrue(os.path.isfile(ofile))


    def test_render_mpi_3d(self):
        # if we don't have ascent, simply return

        info = ascent.mpi.about()
        if info["runtimes/ascent/vtkm/status"] != "enabled":
            print("ascent runtime not enabled, skipping mpi render test")
            return

        obase = "tout_python_ascent_mpi_render_3d"
        ofile = obase + "100.png"
        # clean up old results if they exist
        if MPI.COMM_WORLD.rank == 0 and os.path.isfile(ofile):
            os.remove(ofile)



        # create example mesh using conduit blueprint
        n_mesh = conduit.Node()
        conduit.blueprint.mesh.examples.braid("uniform",
                                              10,
                                              10,
                                              10,
                                              n_mesh)
        # shift data for rank > 1
        x_origin = MPI.COMM_WORLD.rank * 20 - 10;

        n_mesh["state/domain_id"] = MPI.COMM_WORLD.rank
        n_mesh["coordsets/coords/origin/x"] = x_origin
        n_mesh["fields/braid/values"][:] = MPI.COMM_WORLD.rank

        # open ascent
        a = ascent.mpi.Ascent()
        ascent_opts = conduit.Node()
        ascent_opts["mpi_comm"].set(MPI.COMM_WORLD.py2f())
        a.open(ascent_opts)

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

        # use barrier to avoid problems
        # some ranks may finish before file is written
        MPI.COMM_WORLD.Barrier()
        self.assertTrue(os.path.isfile(ofile))


if __name__ == '__main__':
    unittest.main()



