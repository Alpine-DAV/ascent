###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
import json
import argparse
import sys
from collections import OrderedDict

try:
    import textwrap

    textwrap.indent
except AttributeError:  # undefined function (wasn't added until Python 3.3)

    def indent(text, amount, ch=" "):
        padding = amount * ch
        return "".join(padding + line for line in text.splitlines(True))


else:

    def indent(text, amount, ch=" "):
        return textwrap.indent(text, amount * ch)


parser = argparse.ArgumentParser(
    description="Generate sphinx docs for the expressions language."
)
parser.add_argument(
    "functions_json", help="json representation of the function_table conduit node."
)
parser.add_argument(
    "objects_json", help="json representation of the object_table conduit node."
)

#parser.add_argument("output_rst", help="output .rst file")
args = parser.parse_args()

with open('expression_functions.rst', "w+") as docs:
    title = "Expression Functions"
    docs.write(".. _ExpressionFunctions:\n\n")
    docs.write("{}\n{}\n\n".format(title, "=" * len(title)))
    docs.write(
        ".. note:: \n    ``scalar`` is an alias type for either ``int`` or ``double``.\n\n"
    )

    # generate functions
    funcs_title = "Functions"
    docs.write(".. _Ascent Functions Documentation:\n\n")
    docs.write("{}\n{}\n\n".format(funcs_title, "-" * len(funcs_title)))
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
                            sig += "[{}]".format(arg)
                        else:
                            sig += "{}".format(arg)
                        # parameter type
                        params += ":type {}: {}\n".format(
                            arg, func["args"][arg]["type"]
                        )
                        # parameter description
                        if "description" in func["args"][arg]:
                            params += ":param {}: {}\n".format(
                                arg, func["args"][arg]["description"]
                            )
                        else:
                            params += ":param {}:\n".format(arg)
                        if index != len(func["args"]) - 1:
                            sig += ", "
                sig += ")"
                docs.write(".. function:: {}\n\n".format(sig))
                if not "description" in func:
                    sys.exit("Function {} does not have a description.".format(sig))
                docs.write(indent("{}\n\n".format(func["description"]), 4))
                docs.write(indent("{}".format(params), 4))
                docs.write(indent("{}\n\n".format(ret), 4))


with open('expression_objects.rst', "w+") as docs:
    title = "Expression Objects"
    docs.write(".. _ExpressionsObjects:\n\n")
    docs.write("{}\n{}\n\n".format(title, "=" * len(title)))
    #docs.write('

    # generate objects
    objs_title = "Expression Objects"
    docs.write(".. _Ascent Objects Documentation:\n\n")
    docs.write("{}\n{}\n\n".format(objs_title, "-" * len(objs_title)))
    with open(args.objects_json) as json_file:
        objs = json.load(json_file, object_pairs_hook=OrderedDict)
        for obj_name in objs:
            attrs = ""
            for attr_name in objs[obj_name]["attrs"]:
                # object type
                attr = objs[obj_name]["attrs"][attr_name]
                attrs += ":type {}: {}\n".format(attr_name, attr["type"])
                # object description
                if "description" in attr:
                    attrs += ":param {}: {}\n".format(
                        attr_name, attr["description"]
                    )
                else:
                    attrs += ":param {}:\n".format(attr_name)

            docs.write(".. attribute:: {}\n\n".format(obj_name))
            docs.write(indent("{}\n\n".format(attrs), 4))
