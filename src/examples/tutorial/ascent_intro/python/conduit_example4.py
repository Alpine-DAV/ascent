###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import conduit
import numpy as np

#
# Conduit supports zero copy, allowing a Conduit Node to describe and
# point to externally allocated data.
# 
# set_external() is method used to zero copy data into a Node
#
n = conduit.Node()
a1 = np.zeros(10, dtype=np.int32)

a1[0] = 0
a1[1] = 1

for i in range(2,10):
   a1[i] = a1[i-2] + a1[i-1]


# create another array to demo difference 
# between set and set_external
a2 = np.zeros(10, dtype=np.int32) 

a2[0] = 0
a2[1] = 1

for i in range(2,10):
   a2[i] = a2[i-2] + a2[i-1]

n["fib_deep_copy"].set(a1);
n["fib_shallow_copy"].set_external(a2);

a1[-1] = -1
a2[-1] = -1

print(n.to_yaml())