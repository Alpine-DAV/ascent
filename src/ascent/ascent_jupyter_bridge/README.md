# Bridge Kernel

The Bridge Kernel is a Jupyter Kernel for connecting to backends that implement the bridge kernel (bk)
protocol. The bk protocol is a small client-server protocol that emphasizes

* Minimal dependencies
* Easier embedding with MPI
* Secure connections + authenication
* Automatic tunneling

An example MPI Python implementation is available under *server*.

# Getting Started

Bridge Kernel is not on PyPI yet, please clone and install from local source.

```console

spack/bin/spack install -v py-pip ^python@3.6.3
spack/bin/spack activate py-pip
pip install -r requirements.txt
pip install . [--user] [--upgrade]
```

# Writing custom Jupyter Widgets
[Custom Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Custom.html) can be used to extend the Jupyter Notebook with UIs that allow for interactive visualization and analysis of data. This package makes use of these custom widgets is setup to make adding new widgets easy and modular. Those hoping to add their own widgets can look to the existing trackball widget as a template. To add a new module follow these steps:

1. Make a `<widget_name>` directory under `ascent_widgets` this directory will have all the python code for your widgets.
1. In your new directory create a `__init__.py` file to make a package <!-- TODO won't this file need to have duplicate code? -->, also create a `<widget_name>.py` file and define a class that extends `widgets.DOMWidget`. Set the class's `_view_name` to `<_view_name>` and set `_view_module` to `ascent_widgets`. See the `ascent_widgets/trackball/tracball.py` for an example.  <!-- TODO does _view_module_version have to be set? -->
1. Edit `setup.py` and add your new package to the `packages` list.
1. Make a `<widget_name>` directory under `ascent_widgets/js/lib/` and in that folder create your main `<widget_name>.js` file. All your javascript, css, and HTML files should lie in this directory and can be imported via RequireJS. In this file define a RequireJS module that exports an object which extends DOMWidgetView. This object should at least define a "render" method as shown in the [Jupyter Documentation](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Custom.html).
1. Edit the `ascent_widgets/js/lib/index.js` file and import your main javascript file to the by adding it to the `loadedModules` array.
1. Finally create a magic command to display your widget by editting `ascent_jupyter_bridge/kernel.py` and adding your magic to the `widgets` dict along with a callback function that initializes and displays your widget.


More information can be found in [Jupyter's documentation](https://testnb.readthedocs.io/en/latest/examples/Notebook/Distributing%20Jupyter%20Extensions%20as%20Python%20Packages.html).


# Release

Bridge Kernel is released under a BSD-3-Clause license. For more details, please see the
[LICENSE](./LICENSE) file.

`SPDX-License-Identifier: BSD-3-Clause`

`LLNL-CODE-778618`
