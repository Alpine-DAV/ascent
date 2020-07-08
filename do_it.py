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
replay = '/utilities/replay/replay_mpi '
holo = '/utilities/holo_compare/holo '
run_command = 'mpirun -np 8 '

datasets = {'simple_icf': "/Users/larsen30/research/test_builds/holo/datasets/simple_icf/field_dump.cycle_149294.root",
            'triple_q2' : '/Users/larsen30/research/test_builds/holo/datasets/triple_pt_q2q1.cycle_264618.root',
            'triple_q8' : '/Users/larsen30/research/test_builds/holo/datasets/triple_pt_q8q7.cycle_364330.root',
            'triple_q4' : '/Users/larsen30/research/test_builds/holo/datasets/triple_pt_q4q3.cycle_271707.root'}

cameras= { 'simple_icf' : [{'zoom': 1.2},
                           {'zoom': 2.2},
                           {'zoom': 4.2},
                           {'zoom': 6.2}],
          'triple_q2' : [{'zoom' : 1.2},
                         {'zoom' : 2.2}],
          'triple_q4' : [{'zoom' : 1.2},
                         {'zoom' : 2.2}],
          'triple_q8' : [{'zoom' : 1.2},
                         {'zoom' : 2.2}]
         }

refinements = { 'simple_icf' : [2],
                'triple_q2' : [1,2],
                'triple_q4' : [1,2,3],
                'triple_q8' : [1,2,3,4],
              }

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

def gen_lo_actions(output_name, camera):
  actions = ('-\n'
    '  action: \"add_pipelines\"\n'
    '  pipelines:\n'
    '    low_order:\n'
    '      f1:\n'
    '        type: \"project_2d\"\n'
    '        params:\n'
    '          topology: main\n'
    '          image_width: 1024\n'
    '          image_height: 1024\n'
    '          camera:\n')
  for key, value in camera.items():
    actions += '            '+key+': '+str(value)+'\n'
#    '            zoom: 4.2\n'
  actions +=(
    '-\n'
    '  action: \"add_extracts\"\n'
    '  extracts:\n'
    '    low_save:\n'
    '      type: \"relay\"\n'
    '      pipeline: low_order\n'
    '      params:\n'
    '        path: \"'+output_name+'\"\n'
    '        protocol: \"blueprint/mesh/hdf5\"\n')
  return actions

def gen_ho_actions(output_name, camera):
  actions = ('-\n'
    '  action: \"add_pipelines\"\n'
    '  pipelines:\n'
    '    high_order:\n'
    '      f1:\n'
    '        type: \"dray_project_2d\"\n'
    '        params:\n'
    '          image_width: 1024\n'
    '          image_height: 1024\n'
    '          camera:\n')
  for key, value in camera.items():
    actions += '            '+key+': '+str(value)+'\n'
    '            zoom: 4.2\n'
  actions +=(
    '-\n'
    '  action: \"add_extracts\"\n'
    '  extracts:\n'
    '    high_save:\n'
    '      type: \"relay\"\n'
    '      pipeline: high_order\n'
    '      params:\n'
    '        path: \"'+output_name+'\"\n'
    '        protocol: \"blueprint/mesh/hdf5\"\n')
  return actions

def gen_options(refinement):
  options = 'refinement_level : ' + str(refinement) + '\n'
  return options

def write(contents,name):
  f = open(name, 'w')
  f.write(contents)
  f.close()

meta_data = {}
for name,path in datasets.items():
  camera_meta = {}
  for camera in cameras[name]:
    root= ' --root='+path
    act = ' --actions=ascent_actions.yaml'
    cmd = run_command + ascent_build + replay + root + act
    print(camera)
    #do low order stuff
    meta = {}
    low_meta = {}
    for ref in refinements[name]:
      prefix = name+'_'+'low_ref_'+str(ref)+'_c_' + str(camera['zoom'])
      print(prefix)
      write(gen_lo_actions(prefix,camera),'ascent_actions.yaml')
      write(gen_options(ref),'ascent_options.yaml')
      print(cmd)
      low_meta[str(ref)] = prefix
      sexe(cmd,True)
    # do the high order
    prefix = name+'_'+'high_c_' + str(camera['zoom'])
    write(gen_ho_actions(prefix,camera),'ascent_actions.yaml')
    meta['high'] = prefix
    meta['low'] = low_meta
    sexe(cmd,True)
    camera_meta[str(camera['zoom'])] = meta
  meta_data[name] = camera_meta

with open('meta_data.json', 'w') as fp:
    json.dump(meta_data, fp)
