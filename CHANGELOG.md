# Ascent Changelog
Notable changes to Ascent are documented in this file. This changelog started on 8/12/19 and does not document prior changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed
- Issue MPI hang if actions files (yaml or json) fail to parse.




## [0.6.0] - Released 2020-11-06

### Added
- Added support for Devil Ray (high-order) ray tracer
- Added vector operations
  - composite vector (create vector from three scalars)
  - vector component (extract scalar component)
- Allow no refinement for high-order meshes
- Added support for multiple topologies (e.g., volume and particles in the same mesh)
- Added support for AMR Nesting relationships (Blueprint Nestsets)
- Added optional `num_files` parameter to the Relay Extract. See the [Relay Extract Docs](https://ascent.readthedocs.io/en/latest/Actions/Extracts.html#relay) for more details.
- Added an AscentViewer Widget for Jupyter
- Added new CUDA device link logic to help bottle CUDA dependencies for downstream use
- Added support for `exa` prefix style filters

### Changed
- Modified Cinema output so it can be viewed without a webserver
- Removed default behavior of publishing individual vector components when vectors were three separate arrays. This can be achieved by using the vector component filter
- Changed Docker Images to leverage Jupyter lab
- Tutorial updates
- Rendering improvements


## [0.5.1] - Released 2020-01-31

### Added
- Added support to render multiple topologies in the same scene.
- Added a Data Object construct to the main Ascent runtime to easily manage transformations between in-memory mesh representations.

### Fixed
- Issue where cycle was not properly propagated when converting mfem data.
- Cinema issue where zoom was applied additively each cycle to oblivion.
- Cinema issue where cameras were not following the center of the data set.

## [0.5.0] - Released 2019-11-14

### Added

- Added new [Tutorial Content](https://ascent.readthedocs.io/en/latest/Tutorial.html) including C++, Python, and Python-based Jupyter Notebook examples.
- Added docs for [Queries](https://ascent.readthedocs.io/en/latest/Actions/Queries.html) and [Triggers](https://ascent.readthedocs.io/en/latest/Actions/Triggers.html)
- Added a Jupyter Extract that provides interactive Python Notebook access to published mesh data. See the related [Cloverleaf Demo](https://ascent.readthedocs.io/en/latest/Tutorial_CloverLeaf_Demos.html#using-the-jupyter-extract-for-interactive-python-analysis).
- Deprecated the `execute` and `reset` actions. `ascent.execute(actions)` now implicitly resets and execute the Ascent actions. To maintain a degree of backwards compatibility, using `execute` and `reset` are still passable to `ascent.execute(actions)`. Internally, the internal data flow network will only be rebuilt when the current actions differ from the previously executed actions. Note: this only occurs when the Ascent runtime object is persistent between calls to `ascent.execute(actions)`.
- Added support for YAML `ascent_actions` and `ascent_options` files. YAML files are much easier for humans to compose
- Add a relative offset option to the Slice filter.

### Changed

- Several improvements to Ascent's Expression infrastructure.
- Updated our uberenv-based to use a 2019/11 version of spack develop.
- Improved Python error handling and propagation.
- Updated Docker example to build with Jupyter Notebook support.
- Updated to VTK-m 1.5.0 and associated VTK-h.
- Imposed necessary static build constraints for cuda support.


### Fixed
- Several minor bug fixes

[Unreleased]: https://github.com/Alpine-DAV/ascent/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/Alpine-DAV/ascent/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Alpine-DAV/ascent/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Alpine-DAV/ascent/compare/v0.3.0...v0.4.0

