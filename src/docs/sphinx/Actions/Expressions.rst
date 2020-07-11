Ascent Expressions Documentation
================================

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
    
    
.. function:: magnitude(arg1)

    Return the magnitude of the input vector.
    
    :type arg1: vector
    :param arg1:
    :rtype: double
    
    
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
    
    
.. function:: field(arg1)

    Return a mesh field given a its name.
    
    :type arg1: string
    :param arg1:
    :rtype: field
    
    
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
    
    
.. function:: binning(reduction_var, reduction_op, bin_axes, [empty_bin_val], [output])

    Returns a multidimensional data binning.
    
    :type reduction_var: string
    :param reduction_var: The variable being reduced. Either the name of a scalar field on the mesh or one of ``'x'``, ``'y'``, or ``'z'``.
    :type reduction_op: string
    :param reduction_op: The reduction operator to use when   putting values in bins. Available reductions are: 
    
       - cnt: number of elements in a bin 
       - min: minimum value in a bin 
       - max: maximum value in a bin 
       - sum: sum of values in a bin 
       - avg: average of values in a bin 
       - pdf: probability distribution function over all bins 
       - std: standard deviation of values in a bin 
       - var: variance of values in a bin 
       - rms: root mean square of values in a bin
    :type bin_axes: list
    :param bin_axes: List of Axis objects which define the bin axes.
    :type empty_bin_val: scalar
    :param empty_bin_val: The value that empty bins should have. Defaults to 0.
    :type output: string
    :param output: Defaults to ``'none'``. If set to ``'bins'`` a binning with 3 or fewer dimensions will be output as a new topology on the dataset. This is useful for directly visualizing the binning. If set to ``'mesh'`` the bins will be "painted" back onto the original mesh as a new field.
    :rtype: binning
    
    
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
    
    
.. attribute:: value_position

    :type value: double
    :param value:
    :type position: vector
    :param position:
    
    
