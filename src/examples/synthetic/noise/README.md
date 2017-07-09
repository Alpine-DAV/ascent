# Noise.cpp

The noise application generates a synthetic data set based open simplex noise which is a public domain n-dimensional smooth noise function. This is similar to Ken Perlin's simplex noise, but this algorithm does not come with any of the accociated patent issues. The code included here is a *c* port of the java implementation and can be found on [github](https://github.com/smcameron/open-simplex-noise-in-c). The java version can be found [here](https://gist.github.com/KdotJPG/b1270127455a94ac5d19).

The noise application comes with a serial and distributed memory parallel version that will automatically distribute a uniform data set across ranks.

# Options
 - `--time_steps=10` The number of time steps to generate
 - `--time_delta=0.5` The amount of time to advance per time step.
 - `--dims=x,y,z` The total number of cells in the data set
