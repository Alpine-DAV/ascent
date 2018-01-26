#
# file: ascent_tutorial_demo_4_histogram.py
#

import numpy as np

# obtain mpi4py mpi comm
from mpi4py import MPI
comm = MPI.Comm.f2py(ascent_mpi_comm_id())

# get this MPI task's published blueprint data
mesh_data = input()

# get the numpy array for the energy field
e_vals = mesh_data["fields/energy/values"]

# find the data extents of the energy field using mpi
e_min, e_max = e_vals.min(), e_vals.max()

e_min_all = np.zeros(1)
e_max_all = np.zeros(1)

comm.Allreduce(e_min, e_min_all, op=MPI.MIN)
comm.Allreduce(e_max, e_max_all, op=MPI.MAX)

print("min: {} max: {}".format(e_min_all,e_max_all))

# compute bins on global range
bins = np.linspace(e_min_all, e_max_all)

# get histogram counts for local data
hist, bin_edges = np.histogram(e_vals, bins = bins)
print(hist)

# sum histogram counts with MPI to get final histogram

hist_all = np.zeros_like(hist)

comm.Allreduce(hist, hist_all, op=MPI.SUM)

print(hist_all)










