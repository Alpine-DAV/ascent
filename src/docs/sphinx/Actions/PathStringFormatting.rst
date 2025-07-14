.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _PathStringFormatting:

Path String Formatting Overview
===============================

Ascent supports an number of special keyword values that can be used in formatting path names to
generate unique file paths. 

Supported Number Formats
------------------------

The formatting uses c++ generic ``fmt`` formatting as a backend so it will work exactly like
regular string formatting. For reference, we specifically support the following formatting values.

.. note::
    The following examples are formatting the value ``23.657``

.. list-table::
    :widths: 15 75 25 25
    :header-rows: 1

    * - Format
      - Meaning
      - Example Input
      - Example Output
    * - d
      - signed integer
      - 03d
      - 023
    * - i
      - signed integer
      - 03i
      - 023
    * - u
      - unsigned integer
      - 03u
      - 023
    * - f
      - floating-point number (fixed-point)
      - 3.1f
      - 23.7
    * - F 
      - floating-point number (fixed-point, uppercase)
      - 3.1F
      - 23.7
    * - e 
      - decimal exponent notation (scientific, lowercase)
      - 2.2e
      - 2.37e+01
    * - E
      - decimal exponent notation (scientific, uppercase)
      - 2.2E
      - 2.37E+01
    * - g
      - formats as either ``e`` or ``f``, whichever is shorter
      - 2.4g
      - 23.66
    * - G
      - formats as either ``E`` or ``F``, whichever is shorter
      - 2.4G
      - 23.66


Formatting Syntax
-----------------
To use Ascent's keyword formatting, insert the desired keyword in curly braces ``{}`` in your file
path string optionally followed by a colon and the format specifier. 

Syntax: ``{keyword:format}``

If no format specifier is given, the default format for each keyword will be used (see bellow).

Formatting Keywords
-------------------

Several supported keywords can be used in path formatting in Ascent. These are replaced at runtime
with their corresponding values, allowing for dynamic and unique file naming.

Cycle
^^^^^

The current simulation cycle (integer, default format= ``06d``)

This is the default value used for formatting. When standard formatting convention is used in the
path such as ``%03d``, the cycle value will be inserted at runtime. Additionally, in cases where no
formatting is provided, the cycle value will be appended to the end of the path name. 

.. list-table::
    :widths: 50 50 75
    :header-rows: 1

    * - Input Path
      - Final Path Format
      - Resulting Paths
    * - path 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png 
    * - path.png 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png
    * - path_ 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png
    * - path_{cycle} 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png
    * - path_{cycle:} 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png
    * - path_{cycle}.png 
      - path_{cycle:06d}.png 
      - path_000000.png, path_000001.png
    * - path_{cycle:03d} 
      - path_{cycle:03d}.png 
      - path_000.png, path_001.png
    * - path_{cycle:03d} 
      - path_{cycle:03d}.png 
      - path_000.png, path_001.png
    * - path_{cycle:03.1f} 
      - path_{cycle:3.1f}.png 
      - path_0.0.png, path_1.0.png
    * - path_%03d 
      - path_{cycle:03d}.png 
      - path_000.png, path_001.png

Time
^^^^

The simulation time (floating-point, default format= ``g``)

.. note::
    The following examples use ``3.14159`` as the time value

.. list-table::
    :widths: 50 50 75
    :header-rows: 1

    * - Input Path 
      - Final Path Format 
      - Resulting Paths
    * - path_{time} 
      - path_{time:g}.png 
      - path_3.14159.png
    * - path_{time:} 
      - path_{time:g}.png 
      - path_3.14159.png
    * - path_{time}.png 
      - path_{time:g}.png 
      - path_3.14159.png
    * - path_{time:1.3f} 
      - path_{time:1.3f}.png 
      - path_3.142.png
    * - path_{time:3d} 
      - path_{time:3d}.png 
      - path_003.png

Family
^^^^^^

A unique value based on existing files in the output directory (integer, default format= ``06d``)

The family value is used to avoid overwriting existing files in the output directory with a
matching output file pattern, including those from previous runs of Ascent. It is set to one
greater than the maximum detected family value of the matching file pattern, or zero if no
matching files are found.

A minimum family value can be passed to ascent through the options using the ``family_value_seed``
keyword. This will force all family values to be greater than or equal to the seed value.

.. code-block:: cpp
    :caption: Example of setting the family_value_seed

    Node ascent_opts;
    ascent_opts["family_value_seed"] = 200;
    ascent.open(ascent_opts);

.. note::
    For the following examples we will assume that there is an existing file named ``path_0015.png``
    in the output directory.

.. list-table::
    :widths: 50 50 75
    :header-rows: 1

    * - Input Path 
      - Final Path Format 
      - Resulting Paths
    * - path_{family} 
      - path_{family:06d}.png 
      - path_000016.png, path_000017.png
    * - path_{family:} 
      - path_{family:06d}.png 
      - path_000016.png, path_000017.png
    * - path_{family}.png 
      - path_{family:06d}.png 
      - path_000016.png, path_000017.png
    * - path_{family:03d} 
      - path_{family:03d}.png 
      - path_016.png, path_017.png
    * - path_{family:03.1f} 
      - path_{family:03.1f}.png 
      - path_16.0.png, path_17.0.png

.. warning::
    Family value might not behave as expected if running multiple instances of Ascent at the same
    time (e.g. when using parallel in time replay).

Multiple Keyword Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple keywords may be used in any order in the path string to create more descriptive and unique
file path names.

.. list-table::
    :widths: 50 50 75
    :header-rows: 1
    
    * - Input Path 
      - Final Path Format 
      - Resulting Paths
    * - path_{cycle}_{time} 
      - path_{cycle:06d}_{time:g}.png 
      - path_000000_3.14159.png
    * - path_{cycle:03d}_{family:04d} 
      - path_{cycle:03d}_{family:04d}.png 
      - path_000_0016.png
    * - path_time-{time}_cycle-{cycle}_family-{family} 
      - path_time-{time:g}_cycle-{cycle:06d}_family-{family:06d}.png 
      - path_time-3.14159_cycle-000000_family-000016.png

Error Handling and Edge Cases
-----------------------------
 - If an invalid keyword is passed, Ascent will throw a warning and not format that keyword.
 - If an invalid format is passed, Ascent will throw a warning and use the default formatting for that keyword.
 - If a file with the same name already exists and the ``family`` value is not used, the file will be overwritten.

Best Practices
--------------
 - Always use zero-padding for ``cycle`` and ``family`` values to ensure files are sorted correctly.
 - Use the ``family`` keyword when you want to guarantee unique output file names across multiple runs.
 - Use descriptive prefixes and suffixes in your paths for clarity.
