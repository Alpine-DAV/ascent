import sys
import os
import subprocess

from shutil import copyfile
from os.path import join as pjoin


def sexe(cmd):
    print(cmd)
    subprocess.call(cmd,shell=True)

def copy_into_spack(yamls, spack_dir):
    spack_dest = pjoin(spack_dir,"etc/spack/defaults/")
    print("do copy of %s into %s" % (yamls, spack_dest))
    spack_dest_pkgs  = pjoin(spack_dest,"packages.yaml")
    spack_dest_comps = pjoin(spack_dest,"compilers.yaml")
    # remove old
    for f in [spack_dest_pkgs, spack_dest_comps]:
        if os.path.isfile(f):
            os.remove(f)
            print("remove: %s" % f)
        if os.path.isfile(f + ".bkp"):
            os.remove(f + ".bkp")
            print("remove: %s.bkp" % f)
    for f in yamls:
        base = os.path.split(f)[1]
        dst = pjoin(spack_dest,base)
        print("copy: %s to %s" % (f,dst))
        copyfile(f,dst)


def exe_spack_update(spack_dir):
    spack_bin = pjoin(spack_dir,"bin/spack")
    sexe("echo y | " + spack_bin + "  config update packages")
    sexe("echo y | " + spack_bin + "  config update compilers")

def copy_from_spack(yamls, spack_dir):
    dest_dir = os.path.split(yamls[0])[0]
    spack_src = pjoin(spack_dir,"etc/spack/defaults/")
    for f in ["packages.yaml", "compilers.yaml"]:
        chk = pjoin(spack_src,f)
        if os.path.isfile(chk):
            base = os.path.split(f)[1]
            dst = pjoin(dest_dir,base)
            print("copy: %s to %s" % (chk,dst))
            copyfile(chk,dst)


def go_dir(dname,spack_dir):
    for root, dirs, files in os.walk(dname, topdown=False):
        yamls = []
        for name in files:
            if name == "packages.yaml" or name == "compilers.yaml":
                cfg_yaml = os.path.join(root, name)
                yamls.append(cfg_yaml)
        if len(yamls) > 0:
            print(yamls)
            copy_into_spack(yamls,spack_dir)
            exe_spack_update(spack_dir)
            copy_from_spack(yamls,spack_dir)
        for name in dirs:
            go_dir(os.path.join(root, name),spack_dir)


def main():
    src_dir = sys.argv[1]
    spack_dir = sys.argv[2]
    go_dir(src_dir,spack_dir)


if __name__ == "__main__":
    main()