import yaml #pip install --user pyyaml
import pandas as pd
import matplotlib.pyplot as plt

session = []
with open(r'ascent_session.yaml') as file:
  session = yaml.load(file)

binning = session['1d_binning']
cycles = list(binning.keys())
bins = []

# loop through each cycle and grab the bins
for cycle in binning.values():
  bins.append((cycle['attrs']['value']['value']))

# create the coordinate axis using bin centers
x_axis = binning[cycles[0]]['attrs']['bin_axes']['value']['x']
x_min = x_axis['min_val']
x_max = x_axis['max_val']
x_bins = x_axis['num_bins']

x_delta = (x_max - x_min) / float(x_bins)
x_start = x_min + 0.5 * x_delta
x_vals = []
for b in range(0,x_bins):
  x_vals.append(b * x_delta + x_start)

# plot the curve from the last cycle
plt.plot(x_vals, bins[-1]);
plt.xlabel('x position')
plt.ylabel('max radial')
plt.savefig("1d_binning.png")
