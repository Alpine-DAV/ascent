#
# session_to_opac.py
#

import xml.etree.ElementTree as ET
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

def find_color_control_point_list(node):
  """Finds the ColorControlPointList entry in the xml tree."""
  name = str(node.attrib.get('name', node.text))
  if name == "ColorControlPointList":
    # found it!
    print("found it")
    
    print_tree(node)
    return node
  for elem in node:
    res = find_color_control_point_list(elem)
    if res is not None:
        return res

def parse_color_control_points(color_control_point_list_node):
    """Parses the color control points from the xml node and formats them for ascent actions."""
    ascent_color_table = {
        "name": "custom",
        "control_points": []
    }

    for control_point in color_control_point_list_node.iter('Object'):
        if control_point.attrib.get('name') == 'ColorControlPoint':
            color_field = control_point.find("Field[@name='colors']")
            position_field = control_point.find("Field[@name='position']")

            if color_field is not None:
                values = list(map(int, color_field.text.split()))
                if len(values) == 4:  # Ensure the color has RGBA values
                    r, g, b, a = values
                    position = float(position_field.text) if position_field is not None else len(ascent_color_table["control_points"]) / 10.0
                    ascent_color_table["control_points"].append({
                        "type": "rgb",
                        "position": position,
                        "color": [r / 255.0, g / 255.0, b / 255.0]
                    })

    return ascent_color_table


def print_ascent_color_table_yaml(color_table):
    """Prints the Ascent color table in yaml format."""
    print("color_table:")
    print("  control_points:")
    for point in color_table["control_points"]:
      print("    -")
      print("      type: \"%s\"" % point['type'])
      print("      position: %s" % point['position'])
      if point['type'] == 'rgb':
          print("      color: %s" % point['color'])

def write_ascent_color_table_yaml(color_table, output_file):
    """Writes the Ascent color table to a yaml file."""
    with open(output_file, 'w') as yaml_file:
        yaml_file.write("color_table:\n")
        yaml_file.write("  control_points:\n")
        for cp in color_table["control_points"]:
            yaml_file.write("  -\n")
            yaml_file.write("    type: \"%s\"\n" % cp["type"])
            yaml_file.write("    position: %.6f\n" % cp["position"])
            yaml_file.write("    color: %s\n" % cp["color"])

def main():
    if len(sys.argv) != 2:
      print("usage: python3 session_to_color_table {visit_color_table.ct}")
      exit(1)

    session_file = str(sys.argv[1])

    tree = ET.parse(session_file)
    root = tree.getroot()

    color_control_point_list_node = find_color_control_point_list(root)
    if color_control_point_list_node is None:
        print('Warning: failed to find ColorControlPointList entry in %s' % session_file)
    else:
        ascent_color_table = parse_color_control_points(color_control_point_list_node)
        print_ascent_color_table_yaml(ascent_color_table)
        # Write to a YAML file
        output_file = session_file.rsplit('.', 1)[0] + '.yaml'
        write_ascent_color_table_yaml(ascent_color_table, output_file)
main()


