# Ascent Changelog
Notable changes to Ascent are documented in this file. This changelog started on 8/12/19 and does not document prior changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aspires to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Preferred dependency versions for ascent@develop
- conduit@0.8.4
- vtk-m@1.9.0

### Added
- Added pipeline `partition` from Conduit Blueprint
- Added extract `flatten` from Conduit Blueprint
- Added Log base 10 filter. Filter type is `log10`
- Added Log base 2 filter. Filter type is `log2`
- Added Feature Map in the docs. Detailing Devil Ray and VTKh features 
- Added `scripts/build_ascent/build_ascent.sh` a script that demonstrates how to manually build Ascent and its main dependencies
- Added ability to override dimensions for the rendered bounding box around a dataset
- Added CMake option `ENABLE_HIDDEN_VISIBILITY` (default=ON), which controls if hidden visibility is used for private symbols
- Added documentation for how to use ROCm's rocprof profiler for GPUs with Ascent

### Changed
- **The Great Amalgamation** - The VTK-h, Devil Ray, and AP Compositor projects are now developed in Ascent's source instead of separate repos. These external repos for these projects are archived. This reorg simplifies the development and support of these tightly coupled capabilities. Ascent 0.9.0 will be the first release using these internal versions.
- Updated to VTK-m 1.9
- Update docs related to building Ascent.
- Updated to BLT v0.5.2


## [0.8.0] - Released 2022-02-11

### Preferred dependency versions for ascent@0.8.0
- conduit@0.8.2
- dray@0.1.8
- vtk-h@0.8.1
- vtk-m@1.7.1


### Added
- Added OCCA Derived Field Generation support
- Added more math expressions
- Added a time expression
- Added Cinema rendering support for Devil Ray 
- Added `streamline` and `particle_advection` transforms 
- Added history gradient expressions
- Added the ability save named sessions
- Added new options to specify Cinema rendering parameters
- Added the ability save subsets of expression results to session files
- Added the ability to add comments to PNG files that Ascent creates
- Added timings out control option to Ascent (and Flow)
- Added support to render Polygonal nd Polyhedral Meshes 
- Added option to turn of world annotations
- Added FIDES Support

### Fixed
- Fixed a bug where ascent timings files were written out twice
- Fixed a bug where the relay extract protocol was always hdf5, regardless of what was requested
- Various fixes to paraview_ascent_source.py

### Changed
- Python CMake detection logic now prefers Python 3
- Changed Ascent's C-API to use Conduit's C-API object helper methods
- CMake, Spack, and uberenv changes to support newer versions of Cuda, CMake, etc
- Updated to use VTK-m 1.7.0
- Make Ascent Webserver support optional, linked to if Conduit Relay Web support exists
- Simplified the relay extract protocol params, for example can now use `hdf5` instead of `blueprint/mesh/hdf5`
- Updated Spack and Uberenv support for building on Summit

## [0.7.1] - Released 2021-05-20

### Preferred dependency versions for ascent@0.7.1
- conduit@0.7.2
- dray@0.1.6
- vtk-h@0.7.1
- vtk-m@1.5.5


### Added
- Added Data Binning examples to the Ascent Intro Tutorial

### Fixed
- Fixed an issue with the Data Binning bin calculation logic

### Changed
- Updated Ascent to use new conduit, dray, and vtk-h versions


## [0.7.0] - Released 2021-03-19

### Added
- Added partial failure tolerance (i.e., if there are multiple plots the failure of one doesn't prevent the others from rendering)
- Added the ability to use expressions as parameters to filters, e.g., `iso contour value = "(max(field('density')) - min(field('density)) / 2")`
- Added orthogonal projections for scalar images (projecting onto a 2d plane)
- Added a `triangulate` transform
- Added option to build Ascent with only Devil Ray support

### Fixed
- Fixed a MPI hang if actions files (yaml or json) fail to parse
- Fixed several minor issues with saving and reading Mesh Blueprint file sets
- Fixed a field association bug with Data Binning
- Fixed a 2D AMR mesh rendering issue

### Changed
- To better support installs that are relocated on the file system, Cinema database file resources are now compiled into the Ascent library.
- Updated to use Babelflow (1.0.1) and Parallel Merge Tree (1.0.2).

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

[Unreleased]: https://github.com/Alpine-DAV/ascent/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Alpine-DAV/ascent/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Alpine-DAV/ascent/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Alpine-DAV/ascent/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Alpine-DAV/ascent/compare/v0.3.0...v0.4.0
