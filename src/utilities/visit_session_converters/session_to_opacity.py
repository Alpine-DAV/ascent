#
# session_to_opac.py
#

import xml.etree.ElementTree as ET
import math
import sys


def print_tree(node,indent=0):
  """Recursively prints the tree."""
  print (' '*indent + '%s: %s' % (node.tag.title(), node.attrib.get('name', node.text)))
  name = str(node.attrib.get('name', node.text))
  print (' '*indent + '%s' % name)
  indent += 4
  for elem in node:
      print_tree(elem,indent)
  indent -= 4

def find_vol_atts(node):
  name = str(node.attrib.get('name', node.text))
  if name == "VolumeAttributes":
    # found it!
    return node
  for elem in node:
    res = find_vol_atts(elem)
    if res is not None:
        return res

def normalize(x): return x/255.0

def get_field(name, node):
  for field in node.iter('Field'):
    if name == field.attrib.get('name'):
      values = []
      length = field.attrib.get('length')
      field_type = field.attrib.get('type')
      tvals = field.text.split()
      for v in tvals:
        values.append(float(v))
      return values

def print_ascent_opac_yaml(vol_atts):
    field = "freeformOpacity"
    opacity = get_field('freeformOpacity', vol_atts)
    opacity = list(map(normalize, opacity))
    positions = []
    for i in range(0, len(opacity)):
      positions.append(float(i)/float(len(opacity)))
    indent = "  "
    print("control_points:")
    for i in range(0, len(opacity)):
      print(indent + "-")
      print(indent + indent + "type: alpha")
      print(indent + indent + "position: " + str(positions[i]))
      print(indent + indent + "alpha: " + str(opacity[i]))

def main():
    if len(sys.argv) != 2:
      print("usage: python3 session_to_camera {visit.session}")
      exit(1)

    session_file = str(sys.argv[1])

    if 'gui' in session_file:
      print('Warning: gui detected in session file name. Visit saves two files '
            'one for the gui and the other one. I need the other one.')

    tree = ET.parse(session_file)
    root = tree.getroot()

    vol_atts = find_vol_atts(root)
    if vol_atts is None:
      print('Warning: failed to find VolumeAttributes entry in {}'.format(session_file))
    else:
      print_ascent_opac_yaml(vol_atts)

main()


