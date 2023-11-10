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
import platform

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
                             universal_newlines=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = p.communicate()[0]
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
    if len(sys.argv) < 4:
        print("usage: docker_build_and_tag.py {repo_name} {tag_arch} {tag_base} <extra docker build args>")
        sys.exit(-1)
    repo_name = sys.argv[1]
    tag_arch  = sys.argv[2]
    tag_base  = sys.argv[3]

    extra_args = ""
    if len(sys.argv) > 4:
        extra_args = "".join(sys.argv[4:])

    print("repo_name:  {0}".format(repo_name))
    print("tag_arch:   {0}".format(tag_arch))
    print("tag_base:   {0}".format(tag_base))
    print("extra_args: {0}".format(extra_args))

    # remove old source tarball if it exists
    remove_if_exists("{0}.docker.src.tar.gz".format(repo_name))

    # save current working dir so we can get back here
    orig_dir = os.path.abspath(os.getcwd())

    # move to git root dir to get current copy of git source
    rcode, root_dir = sexe("git rev-parse --show-toplevel",ret_output=True)
    root_dir = root_dir.strip()
    print("[repo root dir: {0}]".format(root_dir))
    os.chdir(root_dir)

    # get current copy of the source
    cmd ='python3 package.py {0}'
    sexe(cmd.format(pjoin(orig_dir, "{0}.docker.src.tar.gz".format(repo_name))))

    # change back to orig working dir
    os.chdir(orig_dir)

    # exec docker build to create image
    # tag with date + git hash
    sexe('docker build --build-arg "TAG_ARCH={0}" -t {1}_{2} . {3}'.format(
                                             tag_arch,
                                             tag_base,
                                             gen_docker_tag(),
                                             extra_args))

if __name__ == "__main__":
    main()

