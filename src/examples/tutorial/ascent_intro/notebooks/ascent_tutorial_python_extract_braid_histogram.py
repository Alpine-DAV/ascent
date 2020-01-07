###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
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
