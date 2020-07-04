import os
import sys
import subprocess
import shutil
import socket
import platform
import json
import datetime
import glob

ascent_build = '/Users/larsen30/research/test_builds/holo/ascent/build'
holo = '/utilities/holo_compare/holo '

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

meta_data = ""
with open('meta_data.json') as json_file:
    meta_data = json.load(json_file)

print(meta_data)

for name,values in meta_data.items():
  print(name)
  for zoom,data in values.items():
    high = data['high']
    high_root = glob.glob(high+'*.root')
    print(high_root)
    lows = data['low']
    for ref,low in lows.items():
      prefix = name + '_' + zoom + '_' + ref
      low_root = glob.glob(low+'*.root')
      print(low_root)
      holo_exe = ascent_build + holo + ' '
      cmd = holo_exe + high_root[0] + ' ' + low_root[0] + ' ' + prefix
#      print(cmd)
      sexe(cmd)

