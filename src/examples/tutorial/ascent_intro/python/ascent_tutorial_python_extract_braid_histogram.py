###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
#
# file: ascent_tutorial_python_extract_braid_histogram.py
#
###############################################################################

import numpy as np

# get published blueprint data from the ascent 

# python extract always consumes the multi-domain
# flavor of the mesh blueprint, since we have a 
# single domain mesh, the data we want is the first child

mesh = ascent_data().child(0)

# fetch the numpy array for the braid field values
e_vals = mesh["fields/braid/values"]

# find the data extents of the braid field
e_min, e_max = e_vals.min(), e_vals.max()

# compute bins on extents
bins = np.linspace(e_min, e_max)

# get histogram counts
hist, bin_edges = np.histogram(e_vals, bins = bins)

print("\nEnergy extents: {} {}\n".format(e_min, e_max))
print("Histogram of Energy:\n")
print("Counts:")
print(hist)
print("\nBin Edges:")
print(bin_edges)
print("")

# save our results to a yaml file
hist_results = conduit.Node()
hist_results["hist"] = hist
hist_results["bin_edges"] = bin_edges
hist_results.save("out_py_extract_hist_results.yaml","yaml")
