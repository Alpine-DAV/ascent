###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
"""
file: update_license_header_txt.py
description:
 Simple python script to help create a source file that can be used
 to include the ascent license as a cpp string.
"""

import os
import sys

pattern = {
# c++ style headers
    "hdr":"""//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
""",
    "st": "// "}


def gen_lic_hpp(lic_file,hpp_out):
    lic_txt = open(lic_file).readlines()
    # write the lic prelude, then create var to use in c++
    hpp_f = open(hpp_out,"w")
    hpp_f.write(pattern["hdr"])
    hpp_f.write("// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent\n")
    hpp_f.write("// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and\n")
    hpp_f.write("// other details. No copyright assignment is required to contribute to Ascent.\n")
    hpp_f.write(pattern["hdr"])
    hpp_f.write("\n")
    hpp_f.write("#ifndef ASCENT_LICENSE_TEXT_HPP\n")
    hpp_f.write("#define ASCENT_LICENSE_TEXT_HPP\n\n")
    hpp_f.write("std::string ASCENT_LICENSE_TEXT = ")
    for l in lic_txt:
        ltxt = l.strip().replace("\"","\\\"")
        hpp_f.write("\"%s\\n\"\n" % (ltxt))
    hpp_f.write("\"\";")
    hpp_f.write("\n\n")
    hpp_f.write("#endif\n\n")

if __name__ == "__main__":
    nargs = len(sys.argv)
    modify_files = False
    if nargs < 3:
        print "usage: python generate_cpp_license_header.py "
        print "[new lic] [output file]"
        sys.exit(-1)
    lic_file = sys.argv[1]
    hpp_out  = sys.argv[2]
    gen_lic_hpp(lic_file,hpp_out)

