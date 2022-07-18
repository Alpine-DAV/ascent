.. _ExpressionsObjects:

Expression Objects
==================

.. _Ascent Objects Documentation:

Expression Objects
------------------

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
    
    :param min: Min coordinate of an axis-aligned bounding box (aabb)
    
    :param max: Max coordinate of an axis-aligned bounding box (aabb)
    
    
    
.. attribute:: aabb

    :type min: vector
    :param min:
    :type max: vector
    :param max:
    
    
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
    
    
