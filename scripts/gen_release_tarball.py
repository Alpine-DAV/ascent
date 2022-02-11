# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

import subprocess
import json
from optparse import OptionParser

def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    parser.add_option("--version",
                      dest="ver",
                      default=None,
                      help="version string")
    opts, extras = parser.parse_args()
    # we want a dict b/c
    opts = vars(opts)
    return opts

def shexe(cmd):
    print("[shexe: {0}]".format(cmd))
    subprocess.call(cmd,shell=True)
    
def main():
    opts = parse_args()
    print(json.dumps(opts,indent=2))
    shexe("scripts/git_archive_all.py --prefix ascent-v{0} ascent-v{0}-src-with-blt.tar.gz".format(opts["ver"]))
    shexe("shasum -a 256 ascent-v{0}-src-with-blt.tar.gz".format(opts["ver"]))


if __name__ == "__main__":
    main()
