#
# Helps generate Weak and Strong Scaling Job Scripts
#

import os
import sys
from os.path import join as pjoin

def msub_script_file(opts,tag):
    return  "msub_run_%s_base_%d_i_%d_j_%d_k_%d.sh" % (tag,
                                                       opts["base"],
                                                       opts["i"],
                                                       opts["j"],
                                                       opts["k"])

def run_dir(opts,tag):
    odir = "run_%s_base_%d_i_%d_j_%d_k_%d" % (tag,
                                              opts["base"],
                                              opts["i"],
                                              opts["j"],
                                              opts["k"])
    rdir = pjoin(opts["output_dir"],odir)
    return rdir

def gen_run_dir(opts, tag):
    if not os.path.isdir(opts["output_dir"]):
        os.mkdir(opts["output_dir"])
    rdir = run_dir(opts,tag)
    if(os.path.isdir(rdir)):
        print "WARNING: run dir already exists: %s" % rdir
    else:
        os.mkdir(rdir)
    return rdir

def gen_msub_header_and_run_dir(opts,tag):
    rdir = gen_run_dir(opts,tag)
    res  = "date\n"
    res += "cd " + rdir + "\n"
    return res

def gen_msub_footer(opts,tag):
    res  = "echo $logfile\n"
    res += "date\n"
    return res


def gen_kripke_run(opts):
    # Kripke:
    #
    # script<<"srun -n "<<(i*j*k)<<" "<<dirs->GetAppPath(0)<<" ";
    # script<<"--procs "<<i<<","<<j<<","<<k<<" ";
    # script<<"--zones "<<(i*settings.base)<<","<<(j*settings.base)<<","<<(k*settings.base)<<" ";
    # script<<"--niter 10 --grp 1:1 --legendre 4 --quad 8:4 --dir 4:1\n";
    # script<<"echo $logfile\n";
    
    base = opts["base"] 
    build_dir = opts["build_dir"]
    i = opts["i"]
    j = opts["j"] 
    k = opts["k"]

    kripke_exe = pjoin(build_dir,"examples/proxies/kripke/kripke_par")

    stxt = gen_msub_header_and_run_dir(opts,"kripke")

    stxt += "srun -n %d %s " % ( (i*j*k), kripke_exe)
    stxt += "--procs %d,%d,%d " % (i,j,k)
    stxt += "--zones %d,%d,%d " % ( i*base, j*base, k*base )
    stxt += "--niter 10 --grp 1:1 --legendre 4 --quad 8:4 --dir 4:1\n";

    stxt += gen_msub_footer(opts,"kripke")
    open(msub_script_file(opts,"kripke"),"w").write(stxt)
    return stxt



def gen_lulesh_run(opts):
    # Lulesh
    #   script<<"srun -n "<<(i*i*i)<<" "<<dirs->GetAppPath(1)<<" ";
    #   script<<"-i 10 -p -s "<<settings.base<<"\n";
    #   script<<"echo $logfile\n";
    
    base = opts["base"] 
    build_dir = opts["build_dir"]
    i = opts["i"]
    j = opts["j"] 
    k = opts["k"]

    lulesh_exe = pjoin(build_dir,"examples/proxies/lulesh2.0.3/lulesh_par")
    
    stxt =  gen_msub_header_and_run_dir(opts,"lulesh")
    stxt += "srun -n %s %s " % ( i*i*i,  lulesh_exe)
    stxt += "-i 10 -p -s %d\n" % base
    stxt += gen_msub_footer(opts,"lulesh")

    open(msub_script_file(opts,"lulesh"),"w").write(stxt)
    return stxt
        
def gen_clover_run(opts):
    # Clover:
    #   script<<"     FILE=\"./clover.in\"\n";
    #   script<<"/bin/cat <<EOM >$FILE\n";
    #   script<<"*clover\n";
    #   script<<"state 1 density=0.2 energy=1.0\n";
    #   script<<"state 2 density=1.0 energy=2.5 geometry=cuboid xmin=4.0 xmax=5.0 ymin=4.0 ymax=5.0 zmin=0.0 zmax=10.0\n";
    #   script<<"x_cells="<<i*settings.base<<"\n";
    #   script<<"y_cells="<<j*settings.base<<"\n";
    #   script<<"z_cells="<<k*settings.base<<"\n";
    #   script<<"xmin=0.0\nymin=0.0\nzmin=0.0\nxmax=10.0\nymax=10.0\nzmax=10.0\n";
    #   script<<"initial_timestep=0.04\n";
    #   script<<"max_timestep=0.04\nend_step=10\ntest_problem 1\n";
    #   script<<"visit_frequency=1\n*endclover\n";
    #   script<<"EOM\n";
    #   script<<"srun -n "<<(i*j*k)<<" "<<dirs->GetAppPath(2)<<" \n";
    
    base = opts["base"] 
    build_dir = opts["build_dir"]
    i = opts["i"]
    j = opts["j"] 
    k = opts["k"]
    
    clover_exe = pjoin(build_dir,"examples/proxies/cloverleaf3d-ref/cloverleaf3d_par")

    clover_in  = "*clover\n"
    clover_in += "state 1 density=0.2 energy=1.0\n"
    clover_in += "state 2 density=1.0 energy=2.5 geometry=cuboid xmin=4.0 xmax=5.0 ymin=4.0 ymax=5.0 zmin=0.0 zmax=10.0\n"
    clover_in += "x_cells=%d\n" % (i*base)
    clover_in += "y_cells=%d\n" % (j*base)
    clover_in += "z_cells=%d\n" % (k*base)
    clover_in += "xmin=0.0\nymin=0.0\nzmin=0.0\nxmax=10.0\nymax=10.0\nzmax=10.0\n"
    clover_in += "initial_timestep=0.04\n"
    clover_in += "max_timestep=0.04\nend_step=10\ntest_problem 1\n";
    clover_in += "visit_frequency=1\n"
    clover_in += "*endclover\n";
    
    stxt =  gen_msub_header_and_run_dir(opts,"clover")

    clover_ifile = open(pjoin(run_dir(opts,"clover"),"clover.in"),"w")
    clover_ifile.write(clover_in)
    clover_ifile.close()
    
    stxt += "srun -n %d %s\n" % ( i*j*k, clover_exe)
    
    stxt +=  gen_msub_footer(opts,"clover")
    
    open(msub_script_file(opts,"clover"),"w").write(stxt)
    return stxt



if __name__ == "__main__":
    opts = { "i": 1,
             "j" :1,
             "k": 1,
             "base": 32,
             "build_dir": "/Users/harrison37/Work/alpine/build-debug",
             "output_dir" : "/Users/harrison37/Work/alpine/scripts/scaling/_out"
            }
    gen_kripke_run(opts)
    gen_lulesh_run(opts)
    gen_clover_run(opts)



