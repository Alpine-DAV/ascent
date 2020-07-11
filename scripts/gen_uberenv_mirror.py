###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################
import os
import sys
import subprocess
import glob
import datetime
import shutil

from os.path import join as pjoin


def key_pkgs():
    return  ["ascent", "vtk-h", "vtk-m", "conduit"]

def spec():
    return  "+mfem+dray+babelflow"

def timestamp(t=None,sep="_"):
    """ Creates a timestamp that can easily be included in a filename. """
    if t is None:
        t = datetime.datetime.now()
    #sargs = (t.year,t.month,t.day,t.hour,t.minute,t.second)
    #sbase = "".join(["%04d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d"])
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

def dest_dir():
    res = pjoin("uberenv_mirror","_tarballs")
    if not os.path.isdir(res):
        os.mkdir(res)
    return res

def copy_patches():
    for pkg_name in key_pkgs():
        patches = glob.glob( pjoin("uberenv_libs",
                                   "spack",
                                   "var",
                                   "spack",
                                   "repos",
                                   "builtin",
                                   "packages",
                                   pkg_name,
                                   "*.patch"))
        for patch in patches:
            des_path = pjoin(dest_dir(),pkg_name + "-" + os.path.split(patch)[1])
            print("[copying patch: {0} to {1}]".format(patch,des_path))
            shutil.copyfile(patch,des_path)

def gen_key_tarballs():
    tstamp = timestamp()
    for pkg_name in key_pkgs():
        pkg_tarball = glob.glob(pjoin("uberenv_mirror",pkg_name,"*.tar.gz"))
        if len(pkg_tarball) != 1:
            print("[Error: Could not find mirror tarball of {0}!]".format(pkg_name))
            sys.exit(-1)
        pkg_tarball = pkg_tarball[0]
        src_fname = os.path.split(pkg_tarball)[1]
        pkg_version =  src_fname[len(pkg_name)+1:]
        des_fname =  pkg_name + "-" + tstamp + "-" + pkg_version
        des_path = pjoin(dest_dir(),des_fname)
        print("[copying source tarball: {0} to {1}]".format(pkg_tarball,des_path))
        # copy to dated dir
        shutil.copyfile(pkg_tarball,des_path)

def gen_mirror():
    cmd = 'python {0} --create-mirror --spec="{1}"'.format(pjoin("uberenv","uberenv.py"),
                                                           spec())
    cmd += ' --prefix=uberenv_libs --mirror=uberenv_mirror'
    sexe(cmd)

def main():
    gen_mirror()
    copy_patches()
    gen_key_tarballs()




if __name__ == "__main__":
    main()
