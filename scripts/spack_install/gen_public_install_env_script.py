###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
import os
import sys
import subprocess

from os.path import join as pjoin

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


def spack_exe(spath=None):
    if spath is None:
        to_try = [pjoin("uberenv_libs","spack"), "spack"]
        for p in to_try:
            abs_p = os.path.abspath(p)
            print("[looking for spack directory at: {}]".format(abs_p))
            if os.path.isdir(abs_p):
                print("[FOUND spack directory at: {}]".format(abs_p))
                return os.path.abspath(pjoin(abs_p,"bin","spack"))
        print("[ERROR: failed to find spack directory!]")
        sys.exit(-1)
    else:
        spack_exe = os.path.abspath(pjoin(spath,"bin","spack"))
        if not os.path.isfile(spack_exe):
            print("[ERROR: failed to find spack directory at spath={}]").format(spath)
            sys.exit(-1)
        return spack_exe

def find_pkg(pkg_name, spath = None):
    r,rout = sexe(spack_exe(spath) + " find -p " + pkg_name,ret_output = True)
    print(rout)
    for l in rout.split("\n"):
        print(l)
        lstrip = l.strip()
        if not lstrip == "" and \
           not lstrip.startswith("==>") and  \
           not lstrip.startswith("--"):
            return {"name": pkg_name, "path": l.split()[-1]}
    print("[Warning: failed to find package named '{}', skipping]".format(pkg_name))
    return None

def path_cmd(pkg):
    return('export PATH={}:$PATH\n'.format((pjoin(pkg["path"],"bin"))))

def write_env_script(install_path, pkgs, modules):
    ofile = open("public_env.sh","w")
    ofile.write("#!/bin/bash\n")
    ofile.write("#\n# ascent env helper script\n#\n\n")
    ofile.write("# modules\n")
    for m in modules:
        ofile.write("# {}\n".format(m))
        ofile.write("module load {}\n".format(m))
    ofile.write("\n#spack built packages\n")
    for p in pkgs:
        if not p is None:
            print("[found {} at {}]".format(p["name"],p["path"]))
            ofile.write("# {}\n".format(p["name"]))
            ofile.write(path_cmd(p))
            ofile.write("\n")
    ofile.write("# ascent install path\n")
    ofile.write("export ASCENT_DIR={}/ascent-install\n".format(os.path.abspath(install_path)))
    print("[created {}]".format(os.path.abspath("public_env.sh")))

def main():
    install_path = sys.argv[1]
    modules = []
    if len(sys.argv) > 2:
        modules = sys.argv[2:]
    pkgs = ["cmake", "python"]
    pkgs = [find_pkg(pkg,pjoin(install_path,"spack")) for pkg in pkgs]
    if len(pkgs) > 0:
        write_env_script(install_path, pkgs, modules)
    else:
        print("usage: python gen_public_install.py path modules")

if __name__ == "__main__":
    main()
