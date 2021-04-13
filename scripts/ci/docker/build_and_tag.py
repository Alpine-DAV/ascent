# Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
# Project developers.  See the top-level LICENSE file for dates and other
# details.  No copyright assignment is required to contribute to VisIt.

#
# Helper script that drives our docker build and tag for ci
#

import os
import sys
import subprocess
import shutil
import datetime

from os.path import join as pjoin

def remove_if_exists(path):
    """
    Removes a file system path if it exists.
    """
    if os.path.isfile(path):
        os.remove(path) 
    if os.path.isdir(path):
        shutil.rmtree(path)

def timestamp(t=None,sep="-"):
    """ Creates a timestamp that can easily be included in a filename. """
    if t is None:
        t = datetime.datetime.now()
    sargs = (t.year,t.month,t.day)
    sbase = "".join(["%04d",sep,"%02d",sep,"%02d"])
    return  sbase % sargs

def sexe(cmd,ret_output=False,echo = True):
    """ Helper for executing shell commands. """
    if echo:
        print("[exe: {}]".format(cmd))
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = p.communicate()[0]
        res = res.decode('utf8')
        return p.returncode,res
    else:
        return subprocess.call(cmd,shell=True)

def git_hash():
    """
    Returns the current git repo hash, or UNKNOWN
    """
    res = "UNKNOWN"
    rcode,rout = sexe("git rev-parse HEAD",ret_output=True)
    if rcode == 0:
        res = rout
    return res;

def gen_docker_tag():
    """
    Creates a useful docker tag for the current build.
    """
    ghash = git_hash()
    if ghash != "UNKNOWN":
        ghash = ghash[:6]
    return timestamp() + "-sha" + ghash

def main():
    # get tag-base
    if len(sys.argv) < 2:
        print("usage: build_and_tag.py {tag_base}")
        sys.exit(-1)
    tag_base = sys.argv[1]
    
    # remove old source tarball if it exists
    remove_if_exists("ascent.docker.src.tar.gz")

    # save current working dir so we can get back here
    orig_dir = os.path.abspath(os.getcwd())

    # get current copy of the ascent source
    os.chdir("../../../../")

    # get current copy of the ascent source
    cmd ='python package.py {0}'
    sexe(cmd.format(pjoin(orig_dir, "ascent.docker.src.tar.gz")))

    # change back to orig working dir
    os.chdir(orig_dir)

    # exec docker build to create image
    # note: --squash requires docker runtime with experimental 
    # docker features enabled. It combines all the layers into
    # a more compact final image to save disk space.
    # tag with date + git hash
    sexe('docker build -t {0}_{1} . '.format(tag_base,
                                             gen_docker_tag()))

if __name__ == "__main__":
    main()

