###############################################################################
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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
#
# file: run_ascent_clover_perf_tests.py
#
# purpose:
#  Helper that executes parameterized cloverleaf runs to gather performance
#  data.
#
###############################################################################

import os
import sys
import math
import glob
import shutil
# json instead of yaml b/c it is built-in
import json
import subprocess
from datetime import datetime

from os.path import join as pjoin

# bake in "base_path", the abs path to the dir we are running from
opts = { "base_path": os.path.abspath(os.path.split(__file__)[0])}

def sexe(cmd,ret_output=False,echo = False):
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

def mpiexec_cmd(ntasks):
    return opts["mpiexec_cmd"].format(ntasks)

def clover_cmd():
    return opts["clover_cmd"]

def gen_clover_input_deck(side_ncells=100,
                          ascent_freq=10,
                          end_step=20):
  res = ('*clover\n'
         ' state 1 density=0.2 energy=1.0\n'
         ' state 2 density=1.0 energy=2.5 geometry=cuboid xmin=0.0 xmax=2.0 ymin=0.0 ymax=2.0 zmin=0.0 zmax=2.0\n'
         ' state 3 density=2.0 energy=5.5 geometry=cuboid xmin=4.0 xmax=6.0 ymin=4.0 ymax=6.0 zmin=4.0 zmax=10.0\n'
         ' x_cells=' + str(side_ncells) + '\n'
         ' y_cells=' + str(side_ncells) + '\n'
         ' z_cells=' + str(side_ncells) + '\n'
         ' xmin=0.0\n'
         ' ymin=0.0\n'
         ' zmin=0.0\n'
         ' xmax=10.0\n'
         ' ymax=10.0\n'
         ' zmax=10.0\n'
         ' initial_timestep=0.04\n'
         ' max_timestep=0.04\n'
         ' end_step=' + str(end_step) + ' \n'
         ' test_problem 1\n'
         '\n'
         ' visit_frequency=' + str(ascent_freq) + '\n'
         '*endclover\n')
  return res

def process_results():
    # FUTURE: look for and digest timings yaml files
    yaml_out = glob.glob("vtkh_data_*.yaml")
    if not yaml_out:
      print("[No timing output found!]")
    else:
      print("[Found timing output!]")
      print(yaml_out)


def run_clover(tag, test_opts):
    # setup unique run-dir using tag
    rdir = "_test_" + tag
    if not os.path.isdir(rdir):
        os.mkdir(rdir)
    # change into this dir to run our test
    os.chdir(rdir)
    if "actions_yaml_file" in test_opts.keys():
        actions_file = pjoin(opts["base_path"],test_opts["actions_yaml_file"])
        shutil.copyfile(actions_file, "ascent_actions.yaml")
    if "ntasks" in test_opts.keys():
        ntasks = test_opts["ntasks"]
    else:
        print("[warning: missing ntasks for test {}, defaulting to 2]".format(tag))
        ntasks = 2
    # gen clover input
    open("clover.in","w").write(gen_clover_input_deck())
    # execute clover
    sexe( mpiexec_cmd(ntasks) + " " + clover_cmd())
    # digest results
    process_results()
    # change back to starting dir
    os.chdir("..")

def run_tests():
    for k,v in opts["tests"].items():
        run_clover(k, v)

def post_results():
    res_dirs = glob.glob("_test*")
    print(res_dirs)
    now = datetime.now()
    dir_name = now.strftime('%m.%d.%y-%H.%M')
    sexe('mkdir ' + dir_name)
    sexe('cp -R _test* ' + dir_name)
    tar = dir_name + '.tar.gz'
    sexe('tar cvf '+tar+' '+dir_name)
    clone_success = True
    if not os.path.isdir('ascent_gpu_dashboard'):
      res = sexe('git clone git@github.com:Alpine-DAV/ascent_gpu_dashboard.git')
      print(res)
      if res != 0:
        clone_success = False
    if not clone_success:
      print('Failed to clone dashboard: bailing')
      return
    sexe('mkdir -p ascent_gpu_dashboard/perf_data')
    sexe('cp '+tar+' ascent_gpu_dashboard/perf_data')
    os.chdir('ascent_gpu_dashboard')
    sexe('git add ./perf_data/'+tar)
    sexe('git commit -am \"adding perf data '+dir_name+'\"')
    sexe('git push')
    os.chdir('..')
    sexe('rm -rf ascent_gpu_dashboard')

def usage():
    print("[usage: python run_ascent_clover_perf_tests.py <opts.json>]")

def parse_args():
    # use default opts.json if nothing is passed
    if len(sys.argv) < 2:
        opts_file = "opts.json"
    else:
        opts_file = sys.argv[1]
    opts.update(json.load(open(opts_file)))
    print("Options:")
    print(json.dumps(opts,indent=2))

def main():
    parse_args()
    run_tests()
    post_results()

if __name__ == "__main__":
    main()
