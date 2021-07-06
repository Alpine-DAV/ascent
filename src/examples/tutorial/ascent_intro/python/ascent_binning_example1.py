# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

import conduit
import conduit.blueprint
import ascent
import numpy as np


mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      25,
                                      25,
                                      25,
                                      mesh)


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
queries["q1/params/expression"] = "binning('radial','max', [axis('x',num_bins=20)])";
queries["q1/params/name"] = "1d_binning"

# Create a 2D binning projected onto the x-y plane
queries["q2/params/expression"] = "binning('radial','max', [axis('x',num_bins=20), axis('y',num_bins=20)])";
queries["q2/params/name"] = "2d_binning"

# Create a binning that emulates a line-out, that is, bin all values
# between x = [-1,1], y = [-1,1] along the z-axis in 20 bins.
# The result is a 1x1x20 array
queries["q3/params/expression"] = "binning('radial','max', [axis('x',[-1,1]), axis('y', [-1,1]), axis('z', num_bins=20)])";
queries["q3/params/name"] = "3d_binning"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)



# plot the results 
try:
    import sys
    import yaml #pip install --user pyyaml
    import matplotlib.pyplot as plt
except:
    print("matplotlib not installed, skipping bin result plots")
    sys.exit(0)

# grab info from last execution which includes our binning results
info = conduit.Node()
a.info(info)

#####
# plot the 1d binning result
#####
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
plt.figure(0)
plt.plot(x_vals, bins[-1])
plt.xlabel('x position')
plt.ylabel('max radial')
plt.savefig("1d_binning.png")

#####
# plot the 2d binning result
#####
binning = info.fetch_existing('expressions/2d_binning')
cycles = binning.child_names()
bins = []

# loop through each cycle and grab the bins
for cycle in cycles:
  bins.append(binning[cycle + '/attrs/value/value'])

# extract the values for the unifom bins
def bin_values(axis_name):
  # create the coordinate axis using bin centers
  axis = binning[cycles[0]]['attrs/bin_axes/value/' + axis_name]
  a_min = axis['min_val']
  a_max = axis['max_val']
  a_bins = axis['num_bins']

  a_delta = (a_max - a_min) / float(a_bins)
  a_start = a_min + 0.5 * a_delta

  axis_vals = []
  for b in range(0,a_bins):
    axis_vals.append(b * a_delta + a_start)
  return axis_vals, a_bins

x_vals, x_size = bin_values('x')
y_vals, y_size = bin_values('y')
x, y = np.meshgrid(x_vals, y_vals)
# plot the curve from the last cycle
# Note: values are strided in the order the axes were declared in
# the query, that is the axis listed first varies the fastest
values = np.array(bins[-1]).reshape(x_size, y_size)


# plot the curve from the last cycle
plt.figure(1)
plt.pcolormesh(x, y, values, shading='auto', cmap = 'viridis');
plt.xlabel('x position')
plt.ylabel('y position')
cbar = plt.colorbar()
cbar.set_label('max radial value')
plt.savefig('2d_binning.png')


#####
# plot the 3d binning result
#####
binning = info.fetch_existing('expressions/3d_binning')
cycles = binning.child_names()
bins = []

# loop through each cycle and grab the bins
for cycle in cycles:
  bins.append(binning[cycle + '/attrs/value/value'])

# create the coordinate axis using bin centers
z_axis =  binning[cycles[0]]['attrs/bin_axes/value/z']
z_min = z_axis['min_val']
z_max = z_axis['max_val']
z_bins = z_axis['num_bins']

z_delta = (z_max - z_min) / float(z_bins)
z_start = z_min + 0.5 * z_delta
z_vals = []
for b in range(0,z_bins):
  z_vals.append(b * z_delta + z_start)

# plot the curve from the last cycle
plt.figure(2)
plt.plot(z_vals, bins[-1])
plt.xlabel('z position')
plt.ylabel('max radial')
plt.savefig("3d_binning.png")

# close ascent
a.close()
