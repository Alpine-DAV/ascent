import json
import argparse
from collections import OrderedDict

try:
  import textwrap
  textwrap.indent
except AttributeError:  # undefined function (wasn't added until Python 3.3)
  def indent(text, amount, ch=' '):
    padding = amount * ch
    return ''.join(padding+line for line in text.splitlines(True))
else:
  def indent(text, amount, ch=' '):
    return textwrap.indent(text, amount * ch)

parser = argparse.ArgumentParser(description='Generate sphinx docs for the expressions language.')
parser.add_argument("functions_json", help="json representation of the function_table conduit node.")
parser.add_argument("objects_json", help="json representation of the object_table conduit node.")
parser.add_argument("output_rst", help="output .rst file")
args = parser.parse_args()

with open(args.output_rst, 'w+') as docs:
  title = "Ascent Expressions Documentation"
  docs.write("{}\n{}\n\n".format(title, '='*len(title)))

  # generate functions
  funcs_title = "Functions"
  docs.write("{}\n{}\n\n".format(funcs_title, '-'*len(funcs_title)))
  with open(args.functions_json) as json_file:
    funcs = json.load(json_file, object_pairs_hook=OrderedDict)
    for func_name in funcs:
      # loop over overloads
      for func in funcs[func_name]:
        # name and return type
        sig = "{}(".format(func_name)
        ret = ":rtype: {}\n".format(func["return_type"])
        params = ""
        # function arguments
        if func["args"] is not None:
          for index, arg in enumerate(func["args"]):
            if "optional" in func["args"][arg]:
                sig += "[{}]".format(arg);
            else:
                sig += "{}".format(arg);
            # parameter type
            params += ":type {}: {}\n".format(arg, func["args"][arg]["type"])
            # parameter description
            if "description" in  func["args"][arg]:
              params += ":param {}: {}\n".format(arg, func["args"][arg]["description"])
            else:
              params += ":param {}:\n".format(arg)
            if index != len(func["args"]) - 1:
              sig += ", "
        sig += ")"
        docs.write(".. function:: {}\n\n".format(sig))
        docs.write(indent("{}\n\n".format(func["description"]), 4))
        docs.write(indent("{}".format(params), 4))
        docs.write(indent("{}\n\n".format(ret), 4))

    # generate objects
    objs_title = "Objects"
    docs.write("{}\n{}\n\n".format(objs_title, '-'*len(objs_title)))
    with open(args.objects_json) as json_file:
      objs = json.load(json_file, object_pairs_hook=OrderedDict)
      for obj_name in objs:
        attrs = ""
        for attr_name in objs[obj_name]["attrs"]:
          # object type
          attr = objs[obj_name]["attrs"][attr_name]
          attrs += ":type {}: {}\n".format(attr_name, attr["type"])
          # object description
          if "description" in  attr:
            attrs += ":param {}: {}\n".format(arg_name, attr["description"])
          else:
            attrs += ":param {}:\n".format(attr_name)

        docs.write(".. attribute:: {}\n\n".format(obj_name))
        docs.write(indent("{}\n\n".format(attrs), 4))
