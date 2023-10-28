"""
# Exercise 7 prompts:

Use and modify the code from Query Example 1:

**First**, observe how changing the number of bins alters the entropy of the histogram in Query Example 1; change the number of bins from 128 to 64 and then to 32.

**Second**, add two additional queries like `q2` -- `q3` with 64 bins and `q4` with 32 bins. 

**Third**, plot entropy vs. cycles for each of the three entropy queries. Create arrays to store the entropy as calculated for `q[1-3]` and overlay these entropies on the same plot. 

"""

# ascent + conduit imports
import conduit
import conduit.blueprint
import ascent

import numpy as np
import matplotlib.pyplot as plt

# helpers we use to create tutorial data
from ascent_tutorial_jupyter_utils import tutorial_gyre_example

# cleanup any old results
!./cleanup.sh

# open ascent
a = ascent.Ascent()
a.open()

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_queries"

# declare a queries to ask some questions
queries = add_act["queries"] 

# add a simple query expression (q1)
queries["q1/params/expression"] = "cycle()"
queries["q1/params/name"] = "cycle"

# For the exercise, you can start by changing the input `num_bins=128` to `num_bins=64`
# Then add queries q3 and q4 as below
queries["q2/params/expression"] = "entropy(histogram(field('gyre'), num_bins=128))"
queries["q2/params/name"] = "entropy_of_gyre"

queries["q3/params/expression"] = "entropy(histogram(field('gyre'), num_bins=64))"
queries["q3/params/name"] = "entropy_of_gyre_64bins"

queries["q4/params/expression"] = "entropy(histogram(field('gyre'), num_bins=32))"
queries["q4/params/name"] = "entropy_of_gyre_32bins"

# declare a scene to render the dataset
add_scenes = actions.append()
add_scenes["action"] = "add_scenes"
scenes = add_scenes["scenes"] 
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "gyre"
# Set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_gyre"

# view our full actions tree
print(actions.to_yaml())

# gyre time varying params
nsteps = 10
time = 0.0
delta_time = 0.5

info = conduit.Node()
for step in range(nsteps):
    # call helper that generates a double gyre time varying example mesh.
    # gyre ref :https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
    mesh = tutorial_gyre_example(time)
    
    # update the example cycle
    cycle = 100 + step * 100

    mesh["state/cycle"] = cycle
    print("time: {} cycle: {}".format(time,cycle))
    
    # publish mesh to ascent
    a.publish(mesh)
    
    # update image name
    scenes["s1/image_name"] = "out_gyre_%04d" % step;
    
    # execute the actions
    a.execute(actions)

    # retrieve the info node that contains the query results
    ts_info = conduit.Node()
    a.info(ts_info)
    
    # add to our running info
    info["expressions"].update(ts_info["expressions"])
    
    # update time
    time = time + delta_time
    
# close ascent
a.close()

# create a 2D array with the entropy values from all 
# cycles -- each row corresponds to the values from all
# cycles for a fixed number of bins. Each column has
# entropy values at a particular cycle for 128, 64, and 32 
# bins in the histogram
entropy = np.zeros([3, nsteps])

# For the exercise, we can make gyre a tuple that stores entropy information for all 3 calculations
gyre = info["expressions/entropy_of_gyre"], info["expressions/entropy_of_gyre_64bins"], info["expressions/entropy_of_gyre_32bins"]
# Grab `child_names()` associated with the first element of gyre (this was the original data used in Query Example 1)
cycle_names = gyre[0].child_names()

# transfer conduit data to our summary numpy array
for i in range(3):
    for j in range(gyre[0].number_of_children()):
        entropy[i][j] = gyre[i][j]["value"]

print("Entropy Result")
print(entropy)

# plot the data for each of the 3 entropy arrays
for i in range(3):
    plt.plot(cycle_names, entropy[i])
plt.ylabel('entropy')
plt.xlabel('cycle')
plt.show()