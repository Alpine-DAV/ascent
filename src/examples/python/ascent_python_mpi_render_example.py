###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

"""
 file: ascent_python_mpi_render_example.py

 description:
   Demonstrates using mpi-enabled ascent to render a pseudocolor plot.


 example usage:

 mpiexec -n 2 python ascent_python_mpi_render_example.py

"""

import conduit
import conduit.blueprint
import ascent.mpi

from mpi4py import MPI


# print details about ascent
print(ascent.mpi.about())

# open ascent
a = ascent.mpi.Ascent()
ascent_opts = conduit.Node()
ascent_opts["mpi_comm"].set(MPI.COMM_WORLD.py2f())
a.open(ascent_opts)


# create example mesh using conduit blueprint
n_mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("uniform",
                                      30,
                                      30,
                                      30,
                                      n_mesh)
# shift data for rank > 1
x_origin = MPI.COMM_WORLD.rank * 20 - 10;
n_mesh["coordsets/coords/origin/x"] = x_origin
# provide proper domain id
n_mesh["state/domain_id"] = MPI.COMM_WORLD.rank
# overwrite example field with domain id
n_mesh["fields/braid/values"][:] = MPI.COMM_WORLD.rank

# publish mesh to ascent
a.publish(n_mesh)

# declare a scene to render the dataset
scenes  = conduit.Node()
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"
# Set the output file name (ascent will add ".png")
scenes["s1/image_prefix"] = "out_ascent_render_mpi_3d"

# setup actions to
actions = conduit.Node()
add_act =actions.append()
add_act["action"] = "add_scenes"
add_act["scenes"] = scenes

# execute
a.execute(actions)

# close alpine
a.close()

