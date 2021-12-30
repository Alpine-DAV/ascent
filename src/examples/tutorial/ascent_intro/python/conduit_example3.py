###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import conduit
import numpy as np

#
# Conduit's Node trees hold strings or bitwidth style numeric array leaves
#
# In C++: you can pass raw pointers to numeric arrays or 
#  std::vectors with numeric values
# 
# In python: numpy ndarrays are used for arrays
#
n = conduit.Node()
a = np.zeros(10, dtype=np.int32)

a[0] = 0
a[1] = 1
for i in range(2,10):
   a[i] = a[i-2] + a[i-1]

n["fib"].set(a);
print(n.to_yaml());
