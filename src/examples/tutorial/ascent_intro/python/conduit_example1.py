###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import conduit
import numpy as np

#
# The core of Conduit's data model is `Node` object that 
# holds a dynamic hierarchical key value tree
#
# Here is a simple example that creates
# a key value pair in a Conduit Node:
#
n = conduit.Node()
n["key"] = "value";
print(n.to_yaml())
