# AP Compositor

This is the AP Compositor, a sort-last image compositing library.  This library is designed for use-cases beyond traditional zbuffer image compositing. The intention is to fill gaps in functionality that other image compositing libraries do not support. The AP Compositor is a hybrid-parallel library: with support for both OpenMP (thread-level parallelism) and MPI


# Functionality
We  support two main types of reductions: Radix-k and direct send.
AP Compositor use cases:

 - ZBuffer image compositing
 - Order based compositing: used in compositing volume rendered images of structure meshes
 - Partial compositing: for unstructred volume rendering, no visualization order can be calculated. As rays enter and exit segments of a mesh, local composits are created with a color, pixel id, and depth. These partial composites are sorted and blended in front-to-back order. Partial compositing also supports radiograph data (absorption and absorpton + emission), i.e. an arbitrarty number of channels (energy groups).
  - Scalar rendering: an extention of zbuffer compositing. Instead of a color, scalar image supports attaching binary blobs of data to images, typically containing scalar data from different fields of the mesh. Images are then composited in a traditional manner. This can be used to create deffered rendering data sets for. An example use case is creating Cinema image databases.


# History

The compositing functionality was originally included in the [Ascent](https://github.com/Alpine-DAV/ascent) in-situ visualization library. As time progressed, we separated some functionality into [VTK-h](https://github.com/Alpine-DAV/vtk-h). The compositing functionality is generally useful, so we released it as a standalone library. After a time out on its own, the AP Compositor decided it was time to return to home and amalgamated into Ascent.

The AP Compositor has been tested at scale on the full Sierra supercomputer (16K nodes) at LLNL as well as many other clusters.

# Funding
Work on this library has beed funded by US Department of Energy through the Exascale Computing Project ([ECP](https://www.exascaleproject.org/)) and the [ASC](https://asc.llnl.gov/) program.

# License

AP Compositor is released as part of Ascent, under its BSD-style license.


