# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

#
# plots the 3d binning result from the session file created by
# ascent binning example 1
#

import yaml #pip install --user pyyaml
import matplotlib.pyplot as plt
import numpy as np

session = []
with open(r'ascent_session.yaml') as file:
  session = yaml.load(file)

binning = session['2d_binning']
cycles = list(binning.keys())
bins = []

# loop through each cycle and grab the bins
for cycle in binning.values():
  bins.append((cycle['attrs']['value']['value']))

# extract the values for the unifom bins
def bin_values(axis_name):
  # create the coordinate axis using bin centers
  axis = binning[cycles[0]]['attrs']['bin_axes']['value'][axis_name]
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
plt.pcolormesh(x, y, values, cmap = 'viridis');
plt.xlabel('x position')
plt.ylabel('y position')
cbar = plt.colorbar()
cbar.set_label('max radial value')
plt.savefig('2d_binning.png')
