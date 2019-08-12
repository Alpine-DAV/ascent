# Ascent Noise Example

The noise application generates a synthetic data set based open simplex noise which is a public domain n-dimensional smooth noise function. This is similar to Ken Perlin's simplex noise, but this algorithm does not come with any of the accociated patent issues. The code included here is a *c* port of the java implementation and can be found on [github](https://github.com/smcameron/open-simplex-noise-in-c). The java version can be found [here](https://gist.github.com/KdotJPG/b1270127455a94ac5d19).

The noise application comes with a serial (noise_ser) and distributed memory parallel version (noise_par) that will automatically distribute a uniform data set across ranks.

# Options
 - `--time_steps=10` The number of time steps to generate
 - `--time_delta=0.5` The amount of time to advance per time step.
 - `--dims=x,y,z` The total number of cells in the data set



Sample Runs:

./noise_ser

mpiexec -n 2 ./noise_par

For more info, please visit:

http://ascent.readthedocs.io/en/latest/ExampleIntegrations.html

By default, noise will generate images of a scene with a isosurface and a volume plot.
Additionally, we include the `example_actions.yaml` file containing another set of actions.
To enable the actions decribed in the file, rename it to `ascent_actions.yaml`.
