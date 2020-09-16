.. _Ascent Expressions Documentation:

Ascent Expressions Documentation
================================

.. note:: 
    ``scalar`` is an alias type for either ``int`` or ``double``.

.. _Ascent Functions Documentation:

Functions
---------

.. function:: avg(arg1)

    Return the average of an array.
    
    :type arg1: array
    :param arg1:
    :rtype: double
    
    
.. function:: avg(arg1)

    Return the field average of a mesh variable.
    
    :type arg1: field
    :param arg1:
    :rtype: double
    
    
.. function:: field_nan_count(arg1)

    Return the number  of NaNs in a mesh variable.
    
    :type arg1: field
    :param arg1:
    :rtype: double
    
    
.. function:: field_inf_count(arg1)

    Return the number  of -inf and +inf in a mesh variable.
    
    :type arg1: field
    :param arg1:
    :rtype: double
    
    
.. function:: max(arg1, arg2)

    Return the maximum of two scalars.
    
    :type arg1: scalar
    :param arg1:
    :type arg2: scalar
    :param arg2:
    :rtype: double
    
    
.. function:: max(arg1)

    Return the maximum value from the meshvar. Its position is also stored and is accessible via the `position` function.
    
    :type arg1: field
    :param arg1:
    :rtype: value_position
    
    
.. function:: max(arg1)

    Return the maximum of an array.
    
    :type arg1: array
    :param arg1:
    :rtype: double
    
    
.. function:: max(arg1, arg2)

    Return a derived field that is the max of two fields.
    
    :type arg1: field
    :param arg1:
    :type arg2: scalar
    :param arg2:
    :rtype: jitable
    
    
.. function:: min(arg1)

    Return the minimum value from the meshvar. Its position is also stored and is accessible via the `position` function.
    
    :type arg1: field
    :param arg1:
    :rtype: value_position
    
    
.. function:: min(arg1, arg2)

    Return the minimum of two scalars.
    
    :type arg1: scalar
    :param arg1:
    :type arg2: scalar
    :param arg2:
    :rtype: double
    
    
.. function:: min(arg1)

    Return the minimum of an array.
    
    :type arg1: array
    :param arg1:
    :rtype: double
    
    
.. function:: min(arg1, arg2)

    Return a derived field that is the min of two fields.
    
    :type arg1: field
    :param arg1:
    :type arg2: field
    :param arg2:
    :rtype: jitable
    
    
.. function:: sum(arg1)

    Return the sum of a field.
    
    :type arg1: field
    :param arg1:
    :rtype: double
    
    
.. function:: sum(arg1)

    Return the sum of an array.
    
    :type arg1: array
    :param arg1:
    :rtype: double
    
    
.. function:: cycle()

    Return the current simulation cycle.
    
    :rtype: int
    
    
.. function:: vector(arg1, arg2, arg3)

    Return the 3D position vector for the input value.
    
    :type arg1: scalar
    :param arg1:
    :type arg2: scalar
    :param arg2:
    :type arg3: scalar
    :param arg3:
    :rtype: vector
    
    
.. function:: vector(arg1, arg2, arg3)

    Return a vector field on the mesh.
    
    :type arg1: field
    :param arg1:
    :type arg2: field
    :param arg2:
    :type arg3: field
    :param arg3:
    :rtype: jitable
    
    
.. function:: magnitude(arg1)

    Return the magnitude of the input vector.
    
    :type arg1: vector
    :param arg1:
    :rtype: double
    
    
.. function:: magnitude(vector)

    Return a derived field that is the magnitude of a vector field.
    
    :type vector: field
    :param vector:
    :rtype: jitable
    
    
.. function:: histogram(arg1, [num_bins], [min_val], [max_val])

    Return a histogram of the mesh variable. Return a histogram of the mesh variable.
    
    :type arg1: field
    :param arg1:
    :type num_bins: int
    :param num_bins: defaults to ``256``
    :type min_val: scalar
    :param min_val: defaults to ``min(arg1)``
    :type max_val: scalar
    :param max_val: defaults to ``max(arg1)``
    :rtype: histogram
    
    
.. function:: history(expr_name, [relative_index], [absolute_index])

    As the simulation progresses the expressions   are evaluated repeatedly. The history function allows you to get the value of   previous evaluations. For example, if we want to evaluate the difference   between the original state of the simulation and the current state then we   can use an absolute index of 0 to compare the initial value with the   current value: ``val - history(val, absolute_index=0)``. Another example is if   you want to evaluate the relative change between the previous state and the   current state: ``val - history(val, relative_index=1)``.
    
       .. note:: Exactly one of ``relative_index`` or ``absolute_index`` must be   passed. If the argument name is not specified ``relative_index`` will be   used.
    
    :type expr_name: anytype
    :param expr_name: `expr_name` should be the name of an expression that was evaluated in the past.
    :type relative_index: int
    :param relative_index: The number of evaluations   ago. This should be less than the number of past evaluations. For example,   ``history(pressure, relative_index=1)`` returns the value of pressure one   evaluation ago.
    :type absolute_index: int
    :param absolute_index: The index in the evaluation   history. This should be less than the number of past evaluations. For   example, ``history(pressure, absolute_index=0)`` returns the value of   pressure from the first time it was evaluated.
    :rtype: anytype
    
    
.. function:: entropy(hist)

    Return the Shannon entropy given a histogram of the field.
    
    :type hist: histogram
    :param hist:
    :rtype: double
    
    
.. function:: pdf(hist)

    Return the probability distribution function (pdf) from a histogram.
    
    :type hist: histogram
    :param hist:
    :rtype: histogram
    
    
.. function:: cdf(hist)

    Return the cumulative distribution function (cdf) from a histogram.
    
    :type hist: histogram
    :param hist:
    :rtype: histogram
    
    
.. function:: bin(hist, bin)

    Return the value of the bin at index `bin` of a histogram.
    
    :type hist: histogram
    :param hist:
    :type bin: int
    :param bin:
    :rtype: double
    
    
.. function:: bin(hist, val)

    Return the value of the bin with axis-value `val` on the histogram.
    
    :type hist: histogram
    :param hist:
    :type val: scalar
    :param val:
    :rtype: double
    
    
.. function:: bin(binning, index)

    returns a bin from a binning by index
    
    :type binning: binning
    :param binning:
    :type index: int
    :param index:
    :rtype: bin
    
    
.. function:: field(field_name, [component])

    Return a mesh field given a its name.
    
    :type field_name: string
    :param field_name:
    :type component: string
    :param component: Used to specify a single component if the field is a vector field.
    :rtype: field
    
    
.. function:: topo(arg1)

    Return a mesh topology given a its name.
    
    :type arg1: string
    :param arg1:
    :rtype: topo
    
    
.. function:: point_and_axis(binning, axis, threshold, point, [miss_value], [direction])

    returns the first values in a binning that exceeds a threshold from the given point.
    
    :type binning: binning
    :param binning:
    :type axis: string
    :param axis:
    :type threshold: double
    :param threshold:
    :type point: double
    :param point:
    :type miss_value: scalar
    :param miss_value:
    :type direction: int
    :param direction:
    :rtype: bin
    
    
.. function:: max_from_point(binning, axis, point)

    returns the closest max value from a reference point on an axis
    
    :type binning: binning
    :param binning:
    :type axis: string
    :param axis:
    :type point: double
    :param point:
    :rtype: value_position
    
    
.. function:: quantile(cdf, q, [interpolation])

    Return the `q`-th quantile of the data along   the axis of `cdf`. For example, if `q` is 0.5 the result is the value on the   x-axis which 50% of the data lies below.
    
    :type cdf: histogram
    :param cdf: CDF of a histogram.
    :type q: double
    :param q: Quantile between 0 and 1 inclusive.
    :type interpolation: string
    :param interpolation: Specifies the interpolation   method to use when the quantile lies between two data points ``i < j``: 
    
       - linear (default): ``i + (j - i) * fraction``, where fraction is the   fractional part of the index surrounded by ``i`` and ``j``. 
       - lower: ``i``. 
       - higher: ``j``. 
       - nearest: ``i`` or ``j``, whichever is nearest. 
       - midpoint: ``(i + j) / 2``
    :rtype: double
    
    
.. function:: axis(name, [bins], [min_val], [max_val], [num_bins], [clamp])

    Defines a uniform or rectilinear axis. When used for binning the bins are inclusive on the lower boundary and exclusive on the higher boundary of each bin. Either specify only ``bins`` or a subset of the ``min_val``, ``max_val``, ``num_bins`` options.
    
    :type name: string
    :param name: The name of a scalar field on the mesh or one of ``'x'``, ``'y'``, or ``'z'``.
    :type bins: list
    :param bins: A strictly increasing list of scalars containing the values for each tick. Used to specify a rectilinear axis.
    :type min_val: scalar
    :param min_val: Minimum value of the axis (i.e. the value of the first tick).
    :type max_val: scalar
    :param max_val: Maximum value of the axis (i.e. the value of the last tick).
    :type num_bins: int
    :param num_bins: Number of bins on the axis (i.e. the number of ticks minus 1).
    :type clamp: bool
    :param clamp: Defaults to ``False``. If ``True``, values outside the axis should be put into the bins on the boundaries.
    :rtype: axis
    
    
.. function:: binning(reduction_var, reduction_op, bin_axes, [empty_val], [component], [topo], [assoc])

    Returns a multidimensional data binning.
    
    :type reduction_var: string
    :param reduction_var: The variable being reduced. Either the name of a scalar field on the mesh or one of ``'x'``, ``'y'``, or ``'z'``. ``reduction_var`` should be left empty if ``reduction_op`` is one of ``cnt``, ``pdf``, or ``cdf``.
    :type reduction_op: string
    :param reduction_op: The reduction operator to use when   putting values in bins. Available reductions are: 
    
       - cnt: number of elements in a bin 
       - min: minimum value in a bin 
       - max: maximum value in a bin 
       - sum: sum of values in a bin 
       - avg: average of values in a bin 
       - pdf: probability distribution function 
       - cdf: cumulative distribution function (only supported with 1 axis)
       - std: standard deviation of values in a bin 
       - var: variance of values in a bin 
       - rms: root mean square of values in a bin
    :type bin_axes: list
    :param bin_axes: List of Axis objects which define the bin axes.
    :type empty_val: scalar
    :param empty_val: The value that empty bins should have. Defaults to ``0``.
    :type component: string
    :param component: the component of a vector field to use for the reduction. Example 'x' for a field defined as 'velocity/x'
    :type topo: topo
    :param topo: The topology to bin. Defaults to the topology associated with the bin axes. This topology must have all the fields used for the axes of ``binning``. It only makes sense to specify this when ``bin_axes`` and ``reduction_var`` are a subset of ``x``, ``y``, ``z``.
    :type assoc: topo
    :param assoc: The association of the resultant field. Defaults to the association infered from the bin axes and and reduction variable. It only makes sense to specify this when ``bin_axes`` and ``reduction_var`` are a subset of ``x``, ``y``, ``z``.
    :rtype: binning
    
    
.. function:: paint_binning(binning, [name], [default_val], [topo], [assoc])

    Paints back the bin values onto an existing mesh by binning the elements of the mesh and creating a new field there the value at each element is the value in the bin it falls into.
    
    :type binning: binning
    :param binning: The values in ``binning`` are used to generate the new field.
    :type name: string
    :param name: The name of the new field to be generated. If not specified, a name is automatically generated and the field is treated as a temporary and removed from the dataset when the expression is done executing.
    :type default_val: scalar
    :param default_val: The value given to elements which do not fall into any of the bins. Defaults to ``0``.
    :type topo: topo
    :param topo:  The topology to paint the bin values back onto. Defaults to the topology associated with the bin axes. This topology must have all the fields used for the axes of ``binning``. It only makes sense to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, ``z``. Additionally, it must be specified in this case since there is not enough info to infer the topology assuming there are multiple topologies in the dataset.
    :type assoc: topo
    :param assoc: Defaults to the association infered from the bin axes and and reduction variable. The association of the resultant field. This topology must have all the fields used for the axes of ``binning``. It only makes sense to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, ``z``.
    :rtype: field
    
    
.. function:: binning_mesh(binning, [name])

    A binning with 3 or fewer dimensions will be output as a new element associated field on a new topology on the dataset. This is useful for directly visualizing the binning.
    
    :type binning: binning
    :param binning: The values in ``binning`` are used to generate the new field.
    :type name: string
    :param name: The name of the new field to be generated, the corresponding topology topology and coordinate sets will be named '``name``_topo' and '``name``_coords' respectively. If not specified, a name is automatically generated and the field is treated as a temporary and removed from the dataset when the expression is done executing.
    :rtype: field
    
    
.. function:: sin(arg1)

    Return a derived field that is the sin of a field.
    
    :type arg1: field
    :param arg1:
    :rtype: jitable
    
    
.. function:: abs(arg1)

    Return a derived field that is the absolute value of a field.
    
    :type arg1: field
    :param arg1:
    :rtype: jitable
    
    
.. function:: sqrt(arg1)

    Return a derived field that is the square root value of a field.
    
    :type arg1: field
    :param arg1:
    :rtype: jitable
    
    
.. function:: gradient(field)

    Return a derived field that is the gradient of a field.
    
    :type field: field
    :param field:
    :rtype: jitable
    
    
.. function:: curl(field)

    Return a derived field that is the curl of a vector field.
    
    :type field: field
    :param field:
    :rtype: jitable
    
    
.. function:: derived_field(arg1, [topo], [assoc])

    Cast a scalar to a derived field (type `jitable`).
    
    :type arg1: scalar
    :param arg1: The scalar to be cast to a derived field.
    :type topo: string
    :param topo: The topology to put the derived field onto. The language tries to infer this if not specified.
    :type assoc: string
    :param assoc: The association of the derived field. The language will try to infer this if not specified.
    :rtype: jitable
    
    
.. function:: derived_field(arg1, [topo], [assoc])

    Used to explicitly specify the topology and association of a derived field (e.g. in case it cannot be inferred or needs to be changed).
    
    :type arg1: field
    :param arg1: The scalar to be cast to a derived field.
    :type topo: string
    :param topo: The topology to put the derived field onto. The language tries to infer this if not specified.
    :type assoc: string
    :param assoc: The association of the derived field. The language will try to infer this if not specified.
    :rtype: jitable
    
    
.. function:: binning_value(binning, [default_val], [topo], [assoc])

    Get the value of a vertex or cell in a given binning. In other words, bin the cell and return the value found in that bin of ``binning``.
    
    :type binning: binning
    :param binning: The ``binning`` to lookup values in.
    :type default_val: scalar
    :param default_val: The value given to elements which do not fall into any of the bins. Defaults to ``0``.
    :type topo: topo
    :param topo: The topology to bin. Defaults to the topology associated with the bin axes. This topology must have all the fields used for the axes of ``binning``. It only makes sense to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, ``z``.
    :type assoc: topo
    :param assoc: The association of the resultant field. Defaults to the association infered from the bin axes and and reduction variable. It only makes sense to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, ``z``.
    :rtype: jitable
    
    
.. function:: rand()

    Return a random number between 0 and 1.
    
    :rtype: jitable
    
    
.. _Ascent Objects Documentation:

Objects
-------

.. attribute:: histogram

    :type value: array
    :param value:
    :type min_val: double
    :param min_val:
    :type max_val: double
    :param max_val:
    :type num_bins: int
    :param num_bins:
    :type clamp: bool
    :param clamp:
    
    
.. attribute:: value_position

    :type value: double
    :param value:
    :type position: vector
    :param position:
    
    
.. attribute:: topo

    :type cell: cell
    :param cell: Holds ``jitable`` cell attributes.
    :type vertex: vertex
    :param vertex: Holds ``jitable`` vertex attributes.
    
    
.. attribute:: cell

    :type x: jitable
    :param x: Cell x-coordinate.
    :type y: jitable
    :param y: Cell y-coordinate.
    :type z: jitable
    :param z: Cell z-coordinate.
    :type dx: jitable
    :param dx: Cell dx, only defined for rectilinear topologies.
    :type dy: jitable
    :param dy: Cell dy, only defined for rectilinear topologies.
    :type dz: jitable
    :param dz: Cell dz, only defined for rectilinear topologies.
    :type id: jitable
    :param id: Domain cell id.
    :type volume: jitable
    :param volume: Cell volume, only defined for 3D topologies
    :type area: jitable
    :param area: Cell area, only defined for 2D topologies
    
    
.. attribute:: vertex

    :type x: jitable
    :param x: Vertex x-coordinate.
    :type y: jitable
    :param y: Vertex y-coordinate.
    :type z: jitable
    :param z: Vertex z-coordinate.
    :type id: jitable
    :param id: Domain vertex id.
    
    
.. attribute:: vector

    :type x: double
    :param x:
    :type y: double
    :param y:
    :type z: double
    :param z:
    
    
.. attribute:: bin

    :type min: double
    :param min:
    :type max: double
    :param max:
    :type center: double
    :param center:
    :type value: double
    :param value:
    
    
.. attribute:: jitable

    :type x: jitable
    :param x:
    :type y: jitable
    :param y:
    :type z: jitable
    :param z:
    
    
.. attribute:: field

    :type x: jitable
    :param x:
    :type y: jitable
    :param y:
    :type z: jitable
    :param z:
    
    
