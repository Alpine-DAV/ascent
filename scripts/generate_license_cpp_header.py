###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Strawman. 
# 
# For details, see: http://software.llnl.gov/strawman/.
# 
# Please also read strawman/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################
"""
file: update_license_header_txt.py
description:
 Simple python script to help create a source file that can be used
 to include the strawman license as a cpp string. 
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
    for l in lic_txt:
        hpp_f.write("%s%s" % (pattern["st"],l))
    hpp_f.write(pattern["hdr"])
    hpp_f.write("\n")
    hpp_f.write("#ifndef STRAWMAN_LICENSE_TEXT_HPP\n")
    hpp_f.write("#define STRAWMAN_LICENSE_TEXT_HPP\n\n")
    hpp_f.write("std::string CONDUIT_LICENSE_TEXT = ")
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

