import conduit
import numpy as np

# Example 3
n = conduit.Node()
a = np.zeros(20, dtype=np.int32)

a[0] = 0
a[1] = 1
for i in range(2,20):
   a[i] = a[i-2] + a[i-1]
n["fib"].set(a);
print(n.to_yaml());

