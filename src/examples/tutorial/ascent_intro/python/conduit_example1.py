import conduit
import numpy as np

n = conduit.Node()
n["key"] = "value";
print(n.to_yaml())

