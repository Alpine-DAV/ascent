import conduit
import numpy as np
# Examples 4
# set_external example
n = conduit.Node()
a1 = np.zeros(20, dtype=np.int32)

a1[0] = 0
a1[1] = 1

for i in range(2,20):
   a1[i] = a1[i-2] + a1[i-1]

a2 = np.zeros(20, dtype=np.int32) 

a2[0] = 0
a2[1] = 1

for i in range(2,20):
   a2[i] = a2[i-2] + a2[i-1]

n["fib_deep_copy"].set(a1);
n["fib_shallow_copy"].set_external(a2);

a1[-1] = -1
a2[-1] = -1

print(n.to_yaml())