import conduit
import numpy as np

# Example 2
n = conduit.Node()
n["dir1/dir2/val1"] = 100.5;
print(n.to_yaml()) 

