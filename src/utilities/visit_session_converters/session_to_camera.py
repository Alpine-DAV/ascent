#
# session_to_camera.py
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

def find_view(node):
  name = str(node.attrib.get('name', node.text))
  if name == "View3DAttributes":
    # found it!
    return node
  for elem in node:
    res = find_view(elem)
    if res is not None:
        return res

def print_ascent_view_yaml(view):
    view_d = {}
    for field in view.iter('Field'):
      name = field.attrib.get('name')
      length = field.attrib.get('length')
      is_double = 'double' in field.attrib.get('type')
      if is_double:
        values = []
        tvals = field.text.split()
        for v in tvals:
          values.append(float(v))
        view_d[name] = values

    vn = view_d["viewNormal"];
    mag = math.sqrt(vn[0]*vn[0] + vn[1]*vn[1] + vn[2]*vn[2])
    vn[0] = vn[0] / mag;
    vn[1] = vn[1] / mag;
    vn[2] = vn[2] / mag;
    ps = view_d["parallelScale"][0]
    va = view_d["viewAngle"][0]
    focus = view_d["focus"]
    indent = "  "
    print("camera:")
    pos_str = indent + "position: ["
    pos_str += str(vn[0] * (ps / math.tan(math.pi * va / 360.0)) + focus[0])
    pos_str += ", "
    pos_str += str(vn[1] * (ps / math.tan(math.pi * va / 360.0)) + focus[1])
    pos_str += ", "
    pos_str += str(vn[2] * (ps / math.tan(math.pi * va / 360.0)) + focus[2])
    pos_str += "]"
    print(pos_str)
    print(indent + "look_at: " + str(focus))
    print(indent + "up: " + str(view_d["viewUp"]))
    print(indent + "zoom: " + str(view_d["imageZoom"]))
    print(indent + "fov: " + str(va))

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

    view = find_view(root)
    if view is None:
      print('Warning: failed to find View3DAttributes entry in {}'.format(session_file))
    else:
      print_ascent_view_yaml(view)

main()
