.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _param_schema:

Parameter Schemas for Validation and Surprise Checking
======================================================

The schema validator supports a focused subset of JSON Schema, along with a few Conduit-specific extensions under constraints.

- ``src/libs/flow/flow_schema_validator.hpp``
- ``src/libs/flow/flow_schema_validator.cpp``

The validator is designed to validate *Conduit nodes* against a schema expressed
as another Conduit node.

.. note::
    When writing schemas for this validator, assume:
      - only the fields documented here are supported
      - unsupported JSON Schema keywords are ignored unless explicitly handled by the validator
      - several behaviors differ from standard JSON Schema because they are adapted to Conduit\'s data model

API surface
+++++++++++

The public API is:

- ``bool flow::schema::validate(const conduit::Node &schema, const conduit::Node &input, conduit::Node &info);``
- ``void flow::schema::set_expression_checker(ExpressionCheckFn fn);``

``validate`` returns ``true`` on success. On failure, ``info["errors"]`` is
populated with one or more human-readable error strings.

Basic Schema structure
++++++++++++++++++++++

A schema is itself a Conduit node that describes the expected structure of an input node.

A typical object schema looks like this:

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

Supported keywords
------------------

This section lists every schema keyword the validator checks.

Quick usage pattern
+++++++++++++++++++

In pseudocode, validation looks like:

.. code-block:: c++

  conduit::Node schema; // schema definition
  conduit::Node input;  // node to validate
  conduit::Node info;   // receives errors
  const bool ok = flow::schema::validate(schema, input, info);

If ``ok`` is ``false``, read ``info["errors"]`` (a list of strings) for details.

``type``
  Controls which type-specific rules apply and enforces the input type.

  Supported values:

  - ``"object"``
  - ``"string"``
  - ``"number"``
  - ``"integer"``
  - ``"array"``

  Notes:

  - Unknown ``type`` values are reported as schema errors.
  - For ``"array"``, see the Conduit-specific behavior described above.

  Example:

  .. code-block:: yaml

    # Schema: non-empty string
    type: string
    minLength: 1

``format``
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
  Restricts a *string* input to one of the allowed values.

  - The validator only applies ``enum`` when the input value is a string.
  - The allowed list is expected to be a list of strings in the schema.

  Example:

  .. code-block:: yaml

    type: string
    enum: ["nearest", "linear"]

``allOf``
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

``anyOf``
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

String keywords (``type: "string"``)
++++++++++++++++++++++++++++++++++++

``minLength``
  Requires the string length to be at least this integer value.

  Example:

  .. code-block:: yaml

    type: string
    minLength: 3

``maxLength``
  Requires the string length to be at most this integer value.

  Example:

  .. code-block:: yaml

    type: string
    maxLength: 8

``pattern``
  Requires the string to match the regular expression pattern.

  Notes:

  - The validator uses C++ ``std::regex`` and checks with ``std::regex_match``
    (i.e., it requires a full match, not a substring match).
  - Invalid regex patterns are reported as schema errors.

  Example:

  .. code-block:: yaml

    type: string
    pattern: "^[a-z]+$"

Numeric keywords (``type: "number"`` / ``type: "integer"``)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``minimum``
  Requires the numeric value to be ``>= minimum``.

  Example:

  .. code-block:: yaml

    type: number
    minimum: 0.0

``exclusiveMinimum``
  Requires the numeric value to be ``> exclusiveMinimum``.

  Example:

  .. code-block:: yaml

    type: number
    exclusiveMinimum: 0.0

``maximum``
  Requires the numeric value to be ``<= maximum``.

  Example:

  .. code-block:: yaml

    type: number
    maximum: 1.0

``exclusiveMaximum``
  Requires the numeric value to be ``< exclusiveMaximum``.

  Example:

  .. code-block:: yaml

    type: number
    exclusiveMaximum: 1.0

Object keywords (``type: "object"``)
++++++++++++++++++++++++++++++++++++

``properties``
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
  A list of string field names that must be present on the input object.

  Example:

  .. code-block:: yaml

    type: object
    properties:
      name: {type: string}
    required: [name]

``additionalProperties``
  If present and falsey (``to_int() == 0``), forbids any input object children
  not declared in ``properties``.

  If the keyword is not present, additional properties are allowed.

  Example:

  .. code-block:: yaml

    type: object
    additionalProperties: false
    properties:
      name: {type: string}

Array keywords (``type: "array"``)
++++++++++++++++++++++++++++++++++

``minItems``
  Requires the array length to be at least this integer value.

  Example:

  .. code-block:: yaml

    type: array
    minItems: 1

``maxItems``
  Requires the array length to be at most this integer value.

  Example:

  .. code-block:: yaml

    type: array
    maxItems: 3

``items``
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

Non-standard keywords (``constraints/*``)
+++++++++++++++++++++++++++++++++++++++++

The validator supports several extra keywords under a ``constraints`` object.
These are not standard JSON Schema keywords, but they are validated explicitly.

``constraints/skip``
  If present and truthy (``to_int() != 0``), validation for that schema node is
  skipped entirely (it always succeeds).

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      skip: true

``constraints/forbid``
  For object inputs only. A list of string field names that must *not* be
  present.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      forbid: ["debug", "internal_only"]

``constraints/const``
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

``constraints/not_const``
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

``constraints/exclusiveChildren``
  For object inputs only. Declares a list of mutually-exclusive field names.

  By default, the validator allows *zero or one* of the listed fields to be
  present. If more than one is present, validation fails.

  Example:

  .. code-block:: yaml

    type: object
    constraints:
      exclusiveChildren: ["file", "buffer"]

``constraints/allowNoneInExclusiveGroup``
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
