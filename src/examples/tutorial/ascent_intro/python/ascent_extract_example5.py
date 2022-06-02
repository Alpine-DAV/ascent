###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import conduit.blueprint
import ascent
import numpy as np


mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      25,
                                      25,
                                      25,
                                      mesh)

#
# Ascent provides an embedded Python environment to support
# custom analysis. When using MPI this environment supports 
# distributed-memory processing with one Python interpreter 
# per MPI task.
#
# You can use this environment in Ascent's Python Extract.
#
# In the case you are already using Ascent from Python this may 
# appear a bit odd. Yes, this feature is most useful to 
# provide a Python analysis environment to simulation codes written 
# in other lanauges (C++, C, or Fortran). Reguardless, we can 
# still access it and leverage it from Python.
#
#
# For more detials about the Python extract, see:
# https://ascent.readthedocs.io/en/latest/Actions/Extracts.html#python
#

#
# First, we a histogram calcuation directly in our current 
# Python interpreter and then we will compare results
# with running the same code via a Python Extract.
#

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
hist_results.save("out_hist_results.yaml","yaml")

# Use Ascent to execute the histogram script.
a = ascent.Ascent()
ascent_opts = conduit.Node()
ascent_opts["exceptions"] = "forward"
a.open(ascent_opts)

# publish mesh to ascent
a.publish(mesh);

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_extracts"

# add an extract to execute custom python code
# in `ascent_tutorial_python_extract_braid_histogram.py`

#
# This extract script runs the same histogram code as above,
# but saves the output to `out_py_extract_hist_results.yaml`
# 

extracts = add_act["extracts"]
extracts["e1/type"] = "python"
extracts["e1/params/file"] = "ascent_tutorial_python_extract_braid_histogram.py"


# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

# close ascent
a.close()

#
# Load and compare the extract yaml results,
# they should match our earlier results.
#
hist_extract_results = conduit.Node()
hist_extract_results.load("out_py_extract_hist_results.yaml",protocol="yaml")

diff_info = conduit.Node()
# hist_results is a Node with our earlier results 
if hist_results.diff(hist_extract_results,diff_info):
    print("Extract results differ!")
    print(diff_info.to_yaml())
else:
    print("Extract results match.")
