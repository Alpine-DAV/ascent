# Ascent Changelog
Notable changes to Ascent are documented in this file. This changlog started on 8/12/19 and does not document prior changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

#### General
- Deprecated the `execute` and `reset` actions. `ascent.execute(actions)` now implicitly resets and execute the Ascent actions. To maintain a degree of backwards compatibility, using `execute` and `reset` are still passable to `ascent.execute(actions)`. Internally, the internal data flow network will only be rebuilt when the current actions differ from the previously executed actions. Note: this only occurs when the Ascent runtime object is persistent between calls to `ascent.execute(actions)`.
- Added support for YAML `ascent_actions` and `ascent_options` files. YAML files are much easier for humans to compose.
