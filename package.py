#!/bin/env python
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
#
# file: package.py
#
###############################################################################

import subprocess
import sys
import datetime
import os

from os.path import join as pjoin

def create_package(output_file=None):
    scripts_dir = pjoin(os.path.abspath(os.path.split(__file__)[0]),"scripts")
    pkg_script = pjoin(scripts_dir,"git_archive_all.py");
    repo_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if output_file is None:
         suffix = "tar"
         t = datetime.datetime.now()
         output_file = "%s.%04d.%02d.%02d.%s" % (repo_name,t.year,t.month,t.day,suffix)
    cmd = "python " + pkg_script + " --prefix=ascent " + output_file
    print("[exe: {}]".format(cmd))
    subprocess.call(cmd,shell=True)

if __name__ == "__main__":
    ofile  = None
    if len(sys.argv) > 1:
        ofile  = sys.argv[1]
    create_package(ofile)


