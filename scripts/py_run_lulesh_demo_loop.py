###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import subprocess
import os
from os.path import join as pjoin

def sexe(cmd):
    print "[sexe: %s]" % cmd
    subprocess.call(cmd,shell=True)

def main():
    lulesh_path =pjoin(os.path.dirname(os.path.abspath(__file__)),
                       "../build-debug/examples/lulesh2.0.3/")
    os.chdir(lulesh_path)
    while True:
        sexe("./lulesh_ser -s 8")

if __name__ == "__main__":
    main()
