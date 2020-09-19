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

#printRecur(root)

attr = "VolumeAttributes"

def find_attribute(root, attr_name):
  name = str(root.attrib.get('name', root.text))
  if name == attr_name:
    return root
  for elem in root:
    res = find_attribute(elem, attr_name)
    if res is not None:
      return res

node = find_attribute(root, attr)
print(node)
if type(node) is None:
  print("Attribute %s not found" % attr )
  exit(1)

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

field = "freeformOpacity"
opacity = get_field('freeformOpacity', node)
def normalize(x): return x/255.0
opacity = list(map(normalize, opacity))
positions = []
for i in range(0, len(opacity)):
  positions.append(float(i)/float(len(opacity)))

ctable = {}
cpoints = []
for i in range(0, len(opacity)):
  cpoint = {}
  cpoint['type'] = 'alpha'
  cpoint['position'] = positions[i]
  cpoint['alpha'] = opacity[i]
  cpoints.append(cpoint)

ctable['control_points'] = cpoints
print(yaml.dump(ctable, default_flow_style=False))


#camera = {}
#camera["camera"] = {}
#camera["camera"]["position"] = [0,0,0]
#camera["camera"]["position"][0] = vn[0] * (ps / math.tan(math.pi * va / 360.0)) + focus[0]
#camera["camera"]["position"][1] = vn[1] * (ps / math.tan(math.pi * va / 360.0)) + focus[1]
#camera["camera"]["position"][2] = vn[2] * (ps / math.tan(math.pi * va / 360.0)) + focus[2]
#camera["camera"]["look_at"] = focus
#camera["camera"]["up"] = view_d["viewUp"]
#camera["camera"]["zoom"] = view_d["imageZoom"]
#camera["camera"]["fov"] = va
#print(yaml.dump(camera, default_flow_style=None))
