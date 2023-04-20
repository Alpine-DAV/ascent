###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: ascent_mpi.py
# Purpose: Lazy loads the mpi-enabled ascent interface
#
#  We use lazy loading b/c the ascent and ascent_mpi libraries provide the
#  same symbols, and without this, on some platforms (OSX) importing
#  ascent_python before ascent_mpi_python prevents us from getting the mpi
#  version.
#
###############################################################################


def about():
    try:
        from .ascent_mpi_python import about as ascent_about
        return ascent_about()
    except ImportError:
        raise ImportError('failed to import ascent_mpi_python, was Ascent built with mpi support?')
    return None


def Ascent():
    try:
        from .ascent_mpi_python import Ascent as ascent_obj
        return ascent_obj()
    except ImportError:
        raise ImportError('failed to import ascent_mpi_python, was Ascent built with mpi support?')
    return None

def jupyter_bridge():
    try:
        from ..bridge_kernel.server import jupyter_extract
        return jupyter_extract()
    except ImportError:
        raise ImportError('failed to import ascent_mpi_python, was Ascent built with mpi support?')
    return None





