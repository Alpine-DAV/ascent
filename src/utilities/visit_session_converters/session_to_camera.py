import xml.etree.ElementTree as ET
import math
import yaml
import sys

if len(sys.argv) != 2:
  print("Only one agrument expected: visit session file")
  exit(1)

session_file = str(sys.argv[1])

if 'gui' in session_file:
  print('Warning: gui detected in session file name. Visit saves two files '
        'one for the gui and the other one. I need the other one.')

tree = ET.parse(session_file)
root = tree.getroot()

indent = 0

def printRecur(root):
  """Recursively prints the tree."""
  global indent
  print (' '*indent + '%s: %s' % (root.tag.title(), root.attrib.get('name', root.text)))
  name = str(root.attrib.get('name', root.text))
  print (' '*indent + '%s' % name)
  indent += 4
  for elem in root.getchildren():
      printRecur(elem)
  indent -= 4

view = 0
def find_view(root):
  name = str(root.attrib.get('name', root.text))
  if name == "View3DAttributes":
    global view
    view = root
  for elem in root:
    find_view(elem)

if type(view) is not type(int):
  print("View3DAttributes not found")
  exit(1)
#printRecur(root)
find_view(root)

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
camera = {}
camera["camera"] = {}
camera["camera"]["position"] = [0,0,0]
camera["camera"]["position"][0] = vn[0] * (ps / math.tan(math.pi * va / 360.0)) + focus[0]
camera["camera"]["position"][1] = vn[1] * (ps / math.tan(math.pi * va / 360.0)) + focus[1]
camera["camera"]["position"][2] = vn[2] * (ps / math.tan(math.pi * va / 360.0)) + focus[2]
camera["camera"]["look_at"] = focus
camera["camera"]["up"] = view_d["viewUp"]
camera["camera"]["zoom"] = view_d["imageZoom"]
camera["camera"]["fov"] = va
print(yaml.dump(camera, default_flow_style=None))
