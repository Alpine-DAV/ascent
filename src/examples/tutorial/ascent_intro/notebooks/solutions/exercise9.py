"""
# Exercise 9 prompts:

**First**, swap out “radial” for “braid” in each of the queries for Binning Example 1 such as

```
queries["q1/params/expression"] = "binning('radial','max', [axis('x',num_bins=20)])";
```

Run the resulting code to see how this modifies the final plots.

**Second**, copy and modify the existing code for 1D binning projected along the x axis. Your modified code should 
create a fourth query that 1D bins along the y axis and plot the result. 

**Note** that queries `q2` and `q3` are omitted from the solution for simplicity as they do not change.
"""

# ascent + conduit imports
import conduit
import conduit.blueprint
import ascent

import numpy as np

# cleanup any old results
!./cleanup.sh

# create example mesh using the conduit blueprint braid helper
mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      25,
                                      25,
                                      25,
                                      mesh)

## Data Binning Example

# Use Ascent to bin an input mesh in a few ways
a = ascent.Ascent()

# open ascent
a.open()

# publish mesh to ascent
a.publish(mesh)

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_queries"

# declare a queries to ask some questions
queries = add_act["queries"] 

# Create a 1D binning projected onto the x-axis
queries["q1/params/expression"] = "binning('braid','max', [axis('x',num_bins=20)])";
queries["q1/params/name"] = "1d_binning"

# Create a 2D binning projected onto the x-y plane
queries["q2/params/expression"] = "binning('braid','max', [axis('x',num_bins=20), axis('y',num_bins=20)])";
queries["q2/params/name"] = "2d_binning"

# Create a binning that emulates a line-out, that is, bin all values
# between x = [-1,1], y = [-1,1] along the z-axis in 20 bins.
# The result is a 1x1x20 array
queries["q3/params/expression"] = "binning('braid','max', [axis('x',[-1,1]), axis('y', [-1,1]), axis('z', num_bins=20)])";
queries["q3/params/name"] = "3d_binning"

# Create a 1D binning projected onto the y-axis
queries["q4/params/expression"] = "binning('braid','max', [axis('y',num_bins=20)])";
queries["q4/params/name"] = "1d_binning_y"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

## Plot results

# extra imports for plotting
import yaml
import matplotlib.pyplot as plt

# grab info from last execution which includes our binning results
info = conduit.Node()
a.info(info)

## Plot the 1d binning result projected along x axis

binning = info.fetch_existing('expressions/1d_binning')
cycles = binning.child_names()
bins = []

# loop through each cycle and grab the bins
for cycle in cycles:
  bins.append(binning[cycle + '/attrs/value/value'])

# create the coordinate axis using bin centers
x_axis = binning[cycles[0]]['attrs/bin_axes/value/x']
x_min = x_axis['min_val']
x_max = x_axis['max_val']
x_bins = x_axis['num_bins']

x_delta = (x_max - x_min) / float(x_bins)
x_start = x_min + 0.5 * x_delta
x_vals = []
for b in range(0,x_bins):
  x_vals.append(b * x_delta + x_start)

# plot the curve from the last cycle
plt.plot(x_vals, bins[-1])
plt.xlabel('x position')
plt.ylabel('max braid')

## Plot the 1d binning result projected along y axis

binning = info.fetch_existing('expressions/1d_binning_y')
cycles = binning.child_names()
bins = []

# loop through each cycle and grab the bins
for cycle in cycles:
  bins.append(binning[cycle + '/attrs/value/value'])

# create the coordinate axis using bin centers
y_axis = binning[cycles[0]]['attrs/bin_axes/value/y']
y_min = y_axis['min_val']
y_max = y_axis['max_val']
y_bins = y_axis['num_bins']

y_delta = (y_max - y_min) / float(y_bins)
y_start = y_min + 0.5 * y_delta
y_vals = []
for b in range(0,x_bins):
  y_vals.append(b * y_delta + y_start)

# plot the curve from the last cycle
plt.plot(y_vals, bins[-1])
plt.xlabel('y position')
plt.ylabel('max braid')