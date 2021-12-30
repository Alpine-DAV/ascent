###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import conduit
import numpy as np

#
# Using hierarchical paths imposes a tree structure
#
n = conduit.Node()
n["dir1/dir2/val1"] = 100.5;
print(n.to_yaml()) 
