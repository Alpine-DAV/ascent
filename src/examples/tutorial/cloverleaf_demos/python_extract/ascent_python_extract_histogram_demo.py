###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
#
# file: ascent_tutorial_python_extract_histogram.py
#
###############################################################################

import numpy as np
from mpi4py import MPI

# obtain a mpi4py mpi comm object
comm = MPI.Comm.f2py(ascent_mpi_comm_id())

# get this MPI task's published blueprint data
mesh_data = ascent_data().child(0)

# fetch the numpy array for the energy field values
e_vals = mesh_data["fields/energy/values"]

# find the data extents of the energy field using mpi

# first get local extents
e_min, e_max = e_vals.min(), e_vals.max()

# declare vars for reduce results
e_min_all = np.zeros(1)
e_max_all = np.zeros(1)

# reduce to get global extents
comm.Allreduce(e_min, e_min_all, op=MPI.MIN)
comm.Allreduce(e_max, e_max_all, op=MPI.MAX)

# compute bins on global extents
bins = np.linspace(e_min_all, e_max_all)

# get histogram counts for local data
hist, bin_edges = np.histogram(e_vals[0], bins = bins[0])

# declare var for reduce results
hist_all = np.zeros_like(hist)

# sum histogram counts with MPI to get final histogram
comm.Allreduce(hist, hist_all, op=MPI.SUM)

# print result on mpi task 0
if comm.Get_rank() == 0:
    print("\nEnergy extents: {} {}\n".format(e_min_all[0], e_max_all[0]))
    print("Histogram of Energy:\n")
    print("Counts:")
    print(hist_all)
    print("\nBin Edges:")
    print(bin_edges)
    print("")

