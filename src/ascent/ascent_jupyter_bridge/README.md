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
pip install ipykernel
pip install . [--user] [--upgrade]
```

# Writing custom Jupyter Widgets
TODO
# Release

Bridge Kernel is released under a BSD-3-Clause license. For more details, please see the
[LICENSE](./LICENSE) file.

`SPDX-License-Identifier: BSD-3-Clause`

`LLNL-CODE-778618`
