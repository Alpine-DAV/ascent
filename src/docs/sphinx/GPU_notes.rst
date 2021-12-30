.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################


Ascent GPU Notes
==================

Number of Ranks Per Node
------------------------
On each node, it is important to only launch one MPI rank per GPU.
If your cluster contains multiple GPUs per node, it is important that consecutive ranks be assigned to each node, which is the default behavior of most job schedulers.
By default, each CUDA capable GPU device is queried, and a rank is assigned a device based on the rank number modulo the number of devices.
Collisions could result in a run-time failure or significant delays. 
This default behavior can be overridden. Please see the Ascent options for more details.

