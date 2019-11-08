# Ascent Changelog
Notable changes to Ascent are documented in this file. This changelog started on 8/12/19 and does not document prior changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added new [Tutorial Content](https://ascent.readthedocs.io/en/latest/Tutorial.html) including C++, Python, and Python-based Jupyter Notebook examples.
- Added docs for [Queries](https://ascent.readthedocs.io/en/latest/Actions/Queries.html) and [Triggers](https://ascent.readthedocs.io/en/latest/Actions/Triggers.html)
- Added a Jupyter Extract that provides interactive Python Notebook access to published mesh data. See the related [Cloverleaf Demo](https://ascent.readthedocs.io/en/latest/Tutorial_CloverLeaf_Demos.html#using-the-jupyter-extract-for-interactive-python-analysis).
- Deprecated the `execute` and `reset` actions. `ascent.execute(actions)` now implicitly resets and execute the Ascent actions. To maintain a degree of backwards compatibility, using `execute` and `reset` are still passable to `ascent.execute(actions)`. Internally, the internal data flow network will only be rebuilt when the current actions differ from the previously executed actions. Note: this only occurs when the Ascent runtime object is persistent between calls to `ascent.execute(actions)`.
- Added support for YAML `ascent_actions` and `ascent_options` files. YAML files are much easier for humans to compose

### Changed

- Several improvements to Ascent's Expression infrastructure.
- Updated our uberenv-based to use a 2019/11 version of spack develop.
- Improved Python error handling and propagation.
- Updated Docker example to build with Jupyter Notebook support.
- Updated to VTK-m 1.5.0 and associated VTK-h.
- Imposed necessary static build constraints for cuda support.

 
### Fixed
- Several minor bug fixes 
