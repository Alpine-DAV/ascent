.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _param_schema:

Parameter Schemas for Validation and Surprise Checking
======================================================

The schema validator supports a focused subset of JSON Schema, along with a few Conduit-specific extensions under constraints.

Defined in:

- ``src/libs/flow/flow_schema_validator.hpp``
- ``src/libs/flow/flow_schema_validator.cpp``

The validator is designed to validate *Conduit nodes* against a schema expressed
as another Conduit node.

.. note::
    When writing schemas for this validator, assume:

    - only the fields documented here are supported
    - unsupported JSON Schema keywords are ignored unless explicitly handled by the validator
    - several behaviors differ from standard JSON Schema because they are adapted to Conduit's data model

API Surface
+++++++++++

The public API is:

- ``bool flow::schema::validate(const conduit::Node &schema, const conduit::Node &input, conduit::Node &info);``
- ``void flow::schema::set_expression_checker(ExpressionCheckFn fn);``

``validate`` returns ``true`` on success. On failure, ``info["errors"]`` is
populated with one or more human-readable error strings.


Quick Usage Pattern
+++++++++++++++++++

A schema is itself a Conduit node that describes the expected structure of an input node.

A typical object schema looks like this:

.. code-block:: json

  {
    "type": "object",
    "required": ["name"],
    "properties": {
      "name": {
        "type": "string",
        "minLength": 1
      },
      "count": {
        "type": "integer"
      }
    },
    "additionalProperties": false
  }

In pseudocode, validation looks like:

.. code-block:: c++

  conduit::Node schema; // schema definition
  conduit::Node input;  // node to validate
  conduit::Node info;   // receives errors
  const bool ok = flow::schema::validate(schema, input, info);

If ``ok`` is ``false``, read ``info["errors"]`` (a list of strings) for details.


Supported Keywords
++++++++++++++++++

``type``
--------

  The validator uses ``schema["type"]`` (a string) to decide which validations to
  apply. Supported types are:

  - ``object``: A conduit object node
  - ``string``: A string leaf node
  - ``number``: Any numeric type
  - ``integer``: Any integer type
  - ``array``: a Conduit list, a Conduit object, or a numeric leaf array (a numeric node with ``number_of_elements() >= 1``).

  .. note::
    Empty input nodes (``dtype().is_empty()``) are treated specially:

    - If the schema ``type`` is ``"object"`` and the input is empty, the validator
      validates as if the input were an empty Conduit object.
    - If the schema ``type`` is ``"array"`` and the input is empty, the validator
      validates as if the input were an empty Conduit list.

  .. warning::
    Unknown ``type`` values are reported as schema errors.

  Example:

  .. code-block:: yaml

    # Schema: non-empty string
    type: string
    minLength: 1

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example (type mismatch)</strong></span></summary>

  .. code-block:: yaml

    # Schema
    type: string

    # Input
    42

  .. code-block:: text

    Validation failed at '<root>' (type): type mismatch Expected type 'string'..

  .. raw:: html

    </details>

``format``
----------

  Only the value ``"expression"`` is recognized.

  If the schema contains ``format: "expression"`` and the input value is a
  string, the validator can optionally call an expression checker installed via
  ``flow::schema::set_expression_checker``. If no checker is installed, the
  keyword is treated as a no-op (the validator accepts the value).

  Example:

  .. code-block:: yaml

    type: string
    format: expression

``enum``
--------

  Restricts a *string* input to one of the allowed values.

  - The validator only applies ``enum`` when the input value is a string.
  - The allowed list is expected to be a list of strings in the schema.

  Example:

  .. code-block:: yaml

    type: string
    enum: ["nearest", "linear"]

``allOf``
---------

  Requires that the input validates against *all* subschemas in the array.

  On failure, the validator records a summary message and may add a small number
  of per-option "hint" messages derived from the first error in each failing
  option.

  Example:

  .. code-block:: yaml

    # Require multiple constraints at once.
    allOf:
      - {type: string, minLength: 1}
      - {type: string, pattern: "^[a-z0-9_]+$"}

``oneOf``
---------

  Requires that the input validates against *exactly one* subschema in the
  array.

  On failure, the validator records a summary message and may add a small number
  of per-option "hint" messages derived from the first error in each failing
  option.

  Example:

  .. code-block:: yaml

    # Accept exactly one option; keep ranges non-overlapping to avoid ambiguity.
    oneOf:
      - {type: integer, maximum: 9}
      - {type: integer, minimum: 10}

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example (no matching option; note the per-option hints)</strong></span></summary>

  .. code-block:: yaml

    # Schema
    oneOf:
      - {type: string, pattern: "^a+$"}
      - {type: string, pattern: "^b+$"}

    # Input
    "ccc"

  .. code-block:: text

    Validation failed at '<root>' (oneOf): expected exactly one of 2 schema options to match, but none matched.
        Option 0 hint: Validation failed at '<root>' (pattern): string does not match required pattern '^a+$'.
        Option 1 hint: Validation failed at '<root>' (pattern): string does not match required pattern '^b+$'.

  .. raw:: html

    </details>

``anyOf``
---------

  Requires that the input validates against *at least one* subschema in the
  array.

  On failure, the validator records a summary message and may add a small number
  of per-option "hint" messages derived from the first error in each failing
  option.

  Example:

  .. code-block:: yaml

    # Accept either a string name or an integer id.
    anyOf:
      - {type: string, minLength: 1}
      - {type: integer, minimum: 0}

String Keywords (``type: "string"``)
++++++++++++++++++++++++++++++++++++

``minLength``
-------------

  Requires the string length to be at least this integer value.

  Example:

  .. code-block:: yaml

    type: string
    minLength: 3

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example</strong></span></summary>

  .. code-block:: yaml

    # Input
    "ab"

  .. code-block:: text

    Validation failed at '<root>' (minLength): string is too short: length is 2. Expected string length >= 3.

  .. raw:: html

    </details>

``maxLength``
-------------

  Requires the string length to be at most this integer value.

  Example:

  .. code-block:: yaml

    type: string
    maxLength: 8

``pattern``
-----------

  Requires the string to match the regular expression pattern.

.. note::

  - The validator uses C++ ``std::regex`` and checks with ``std::regex_search``
    (i.e., the pattern may match any substring; it is not implicitly anchored).
    To require a full-string match, add anchors (e.g., ``^...$``).
  - Invalid regex patterns are reported as schema errors.

  Example:

  .. code-block:: yaml

    type: string
    pattern: "^[a-z]+$"

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example (pattern mismatch)</strong></span></summary>

  .. code-block:: yaml

    # Input
    "Abc"

  .. code-block:: text

    Validation failed at '<root>' (pattern): string does not match required pattern '^[a-z]+$'.

  .. raw:: html

    </details>

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example (schema error from invalid regex)</strong></span></summary>

  .. code-block:: yaml

    # Schema
    type: string
    pattern: "["

    # Input
    "anything"

  .. code-block:: text

    Schema error near '<root>' (pattern): invalid regex pattern '['.

  .. raw:: html

    </details>

Numeric Keywords (``type: "number"`` / ``type: "integer"``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``minimum``
-----------

  Requires the numeric value to be ``>= minimum``.

  Example:

  .. code-block:: yaml

    type: number
    minimum: 0.0

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example</strong></span></summary>

  .. code-block:: yaml

    # Input
    -1.0

  .. code-block:: text

    Validation failed at '<root>' (minimum): -1.000000 is below the allowed minimum. Expected number >= 0.

  .. raw:: html

    </details>

``exclusiveMinimum``
--------------------

  Requires the numeric value to be ``> exclusiveMinimum``.

  Example:

  .. code-block:: yaml

    type: number
    exclusiveMinimum: 0.0

``maximum``
-----------

  Requires the numeric value to be ``<= maximum``.

  Example:

  .. code-block:: yaml

    type: number
    maximum: 1.0

``exclusiveMaximum``
--------------------

  Requires the numeric value to be ``< exclusiveMaximum``.

  Example:

  .. code-block:: yaml

    type: number
    exclusiveMaximum: 1.0

Object Keywords (``type: "object"``)
++++++++++++++++++++++++++++++++++++

``properties``
--------------

  Declares named subschemas for object children.

  The validator only validates properties that are present in the input. Missing
  properties are not an error unless required by ``required``.

  Example (used below with ``required`` and ``additionalProperties``):

  .. code-block:: yaml

    type: object
    additionalProperties: false
    properties:
      name: {type: string, minLength: 1}
      count: {type: integer, minimum: 0}
    required: [name, count]

``required``
------------

  A list of string field names that must be present on the input object.

  Example:

  .. code-block:: yaml

    type: object
    properties:
      name: {type: string}
    required: [name]

``additionalProperties``
------------------------

  If present and falsey (``to_int() == 0``), forbids any input object children
  not declared in ``properties``. This is equivalent to enforcing "surprise
  checking" against the schema.

  If the keyword is not present, additional properties are allowed.

  Example:

  .. code-block:: yaml

    type: object
    additionalProperties: false
    properties:
      name: {type: string}

  .. raw:: html

    <details class="schema-failure"><summary><span style="color:#b94a48;"><strong>Validation Failure Example</strong></span></summary>

  .. code-block:: yaml

    # Input
    name: "ok"
    debug: "nope"

  .. code-block:: text

    Validation failed at 'debug' (additionalProperties): unexpected additional field is not allowed here.

  .. raw:: html

    </details>

Array Keywords (``type: "array"``)
++++++++++++++++++++++++++++++++++

``minItems``
------------

  Requires the array length to be at least this integer value.

  Example:

  .. code-block:: yaml

    type: array
    minItems: 1

``maxItems``
------------

  Requires the array length to be at most this integer value.

  Example:

  .. code-block:: yaml

    type: array
    maxItems: 3

``items``
---------

  A subschema applied to each element of the input when the input is represented
  as a Conduit list or object.

  Notes:

  - When the input is a numeric leaf array, ``items`` is not applied (only
    ``minItems``/``maxItems`` are checked).

  Example:

  .. code-block:: yaml

    type: array
    minItems: 1
    maxItems: 3
    items: {type: integer, minimum: 0}

Non-Standard Keywords (``constraints/*``)
+++++++++++++++++++++++++++++++++++++++++

The validator supports several extra keywords under a ``constraints`` object.
These are not standard JSON Schema keywords, but they allow for more flexible
and explicit validation schemas.

``constraints/skip``
--------------------

  If present and truthy (``to_int() != 0``), validation for that schema node is
  skipped entirely (it always succeeds).

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      skip: true


  .. note::
    This additional feature allows for schemas to be defined without being explicitly validated
    against or surprise checked. This allows for subsets of a schema to be validated while the
    remainder of the schema can be ignored or validated at a later time.

``constraints/forbid``
----------------------

  For object inputs only. A list of string field names that must *not* be
  present.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      forbid: ["debug", "internal_only"]

  .. note::
    Standard JSON Schema can express "forbidden properties", but it is verbose
    (typically involving nested ``not`` + ``required`` combinations). This keyword
    is a compact way to enforce deprecations, feature gating, or to block internal
    knobs from user-facing schemas while keeping the rest of the object open to
    evolution.

``constraints/const``
---------------------

  Requires the input to exactly match a constant value.

  Supported constant types in the schema are:

  - string
  - integer
  - number

  Unsupported constant types are reported as schema errors.

  Example:

  .. code-block:: yaml

    # Require a fixed string "v1"
    constraints: {const: "v1"}


  .. note::
    Standard JSON Schema has a top-level ``const`` keyword, but this validator
    nests some non-standard rules under ``constraints``. Keeping const here allows
    schemas to group "extra constraints" together and keeps the validator's
    behavior explicit and easy to scan in Conduit/YAML.

``constraints/not_const``
-------------------------

  For object inputs only. Forbids specific constant values for specific fields.

  The schema value is expected to be an object whose children are field names,
  where each field maps to a forbidden constant value (string, integer, or
  number). If the input contains that field with the same value, validation
  fails.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      not_const:
        mode: "unsafe"
        retries: 0

  .. note::
    This is a concise way to ban specific values (e.g., ``mode: "unsafe"``
    or ``retries: 0``) without rewriting the entire field schema or adding complex
    boolean logic. In standard JSON Schema this typically requires multiple
    ``not`` clauses that can be harder to read and maintain.

``constraints/dependencies``
  For object inputs only. Declares field dependencies.

  The schema value is expected to be an object mapping a *trigger field* name to
  a list of *required field* names. If the trigger field is present in the
  input, all required fields must also be present.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      dependencies:
        output_path: ["output_protocol"]

  .. note::
    JSON Schema dependency tracking varries between versions of JSON schema. This validator
    elected to match the language used in Draft 7. Similarly to ``constraints/const``, 
    standard JSON Schema generally has dependencies at the tope level, but this validator
    elects to store them under constraints. Keeping const here allows
    schemas to group "extra constraints" together and keeps the validator's
    behavior explicit and easy to scan in Conduit/YAML.

``constraints/exclusiveChildren``
---------------------------------

  For object inputs only. Declares a list of mutually-exclusive field names.

  By default, the validator allows *zero or one* of the listed fields to be
  present. If more than one is present, validation fails.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      exclusiveChildren: ["file", "buffer"]

  .. note::
    Mutually-exclusive property groups are common in parameter schemas ("choose
    one of these ways to specify the input"). Standard JSON Schema can encode this
    with ``oneOf`` and multiple ``required``/``not`` combinations, but the result
    is verbose and difficult to maintain as the option set grows. This keyword
    provides a consise and simple approach to this functionality.

``constraints/allowNoneInExclusiveGroup``
-----------------------------------------

  Only used with ``constraints/exclusiveChildren``.

  If present and falsey (``to_int() == 0``), the validator requires *exactly
  one* of the mutually-exclusive fields to be present (it becomes an error if
  none are present).

  Example (require exactly one instead of zero-or-one):

  .. code-block:: yaml

    type: object
    constraints:
      exclusiveChildren: ["file", "buffer"]
      allowNoneInExclusiveGroup: false

  .. note::
    Many schemas want "exactly one" while others want "zero or one" (optional
    selection). This keyword extends the ``constraints/exclusiveChildren`` keyword
    and allows schemas to toggle between the two behaviors.
