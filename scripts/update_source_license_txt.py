###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Alpine. 
# 
# For details, see: http://software.llnl.gov/alpine/.
# 
# Please also read alpine/LICENSE
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
file: update_source_license_txt.py
description:
 Simple python script to help with update license header text in files  
 throughout the source tree.
 
 usage: python update_source_license_txt.py [old lic] [new lic] [exec]"

"""

import os
import sys

patterns = {
# c++ style headers
"cpp":{
    "hdr":"""
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
""",
    "st": "// "},
# pound prefixed style headers (python + cmake)
"pnd":{
    "hdr":"""
###############################################################################
""",
    "st":"# "},
# ReStructured Text style headers
"rst":{
    "hdr":"""
.. ############################################################################
""",
    "st":".. # "},
# Fortran Text style headers
"fortran":{
    "hdr":"""
!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
""",
    "st":"!* "}}


def gen_formatted(lic_txt,hdr,st):
    res = hdr
    keep = True
    for l in lic_txt:
        if l.count("Additional BSD Notice") > 0:
            keep = False
        if keep:
            res += st + l
    res += hdr
    return res

def update_lic(lic_file_old,lic_file_new,modify_files):
    old_lic_txt   = open(lic_file_old).readlines()
    lic_short_txt = open(lic_file_new).readlines()
    for k,v in patterns.items():
        v["full"] = gen_formatted(old_lic_txt,
                                  v["hdr"][1:],
                                  v["st"])    
    all_files = []
    for root_path in ["../src",
                      "../scripts",
                      "../host-configs",
                      "../config-build.sh",
                      "../bootstrap-env.sh",
                      "../package.py"]:

        if os.path.isdir(root_path):
            for dirpath, dnames, fnames in os.walk(root_path):
                for f in fnames:
                    full = os.path.abspath(os.path.join(dirpath, f))
                    all_files.append(full)
        else:
            all_files.append(os.path.abspath(root_path))
    print all_files
    up_cnt    = {}
    updated   = []
    for f in all_files:
        curr_txt =  open(f).read()
        for k,v in patterns.items():
            full_old_lic = v["full"]
            cnt = curr_txt.count(full_old_lic)
            if cnt > 0:
                print "[begin update to %s (style = %s)]" % (f,k)
                print "[old - %s]" % f
                print curr_txt
                new_txt = curr_txt.replace(full_old_lic,
                                           gen_formatted(lic_short_txt,
                                                         v["hdr"][1:],
                                                         v["st"]))
                print "[new - %s]" % f
                print new_txt
                print "[end update to %s]" % f
                updated.append(f)
                if not f in up_cnt.keys():
                    up_cnt[f] = 0
                up_cnt[f] += 1
                if modify_files:
                    open(f,"w").write(new_txt)
    print updated, len(updated)
    for k,v in up_cnt.items():
        print k,v


if __name__ == "__main__":
    nargs = len(sys.argv)
    modify_files = False
    if nargs < 3:
        print "usage: python update_license_headers.py "
        print "[old lic] [new lic] [exec]"
        sys.exit(-1)
    lic_file_old = sys.argv[1]
    lic_file_new = sys.argv[2]
    modify_files =  "--exec" in sys.argv
    update_lic(lic_file_old,lic_file_new,modify_files)

