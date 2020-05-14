###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
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
 file: gen_img_compare_html_report.py
 description: 
    Generates a html file that collects image diff results from ascent's tests

"""
import json
import glob
import os
import shutil

from os.path import join as pjoin

def output_dir():
    return "_output"


def file_name(fpath):
    return os.path.split(fpath)[1]

def baseline_file_path(fname):
    baseline_dir = pjoin("..","..","src","tests","baseline_images")
    return pjoin(baseline_dir,fname)

def find_img_compare_results():
    return glob.glob(pjoin(output_dir(),"*_img_compare_results.json"))

def gen_html_report():
    ofile = open(pjoin("_output","tout_img_report.html"),"w")
    ofile.write("<table border=1>\n")
    ofile.write("<tr>\n")
    ofile.write("<td>[INFO]</td>\n")
    ofile.write("<td>[CURRENT]</td>\n")
    ofile.write("<td>[BASELINE]</td>\n")
    ofile.write("<td>[DIFF]</td>\n")
    ofile.write("</tr>\n")
    for res in find_img_compare_results():
        v = json.load(open(res))
        ofile.write("<tr>\n")
        ofile.write("<td>\n")
        ofile.write("case: {0} <br>\n".format(file_name(v["test_file"]["path"])))
        ofile.write("pass: {0} <br>\n".format(v["pass"]))
        for k in ["dims_match","percent_diff","tolerance"]:
           ofile.write("{0} = {1} <br>\n".format(k,v[k]))
        ofile.write("</td>\n")
        if v["test_file"]["exists"] == "true":
            ofile.write('<td><img width="200" src="{0}"><br>current</td>\n'.format(file_name(v["test_file"]["path"])))
        else:
            ofile.write("<td> TEST IMAGE MISSING!</td>\n")
        if v["baseline_file"]["exists"] == "true":
            baseline_img_src = baseline_file_path(file_name(v["baseline_file"]["path"]))
            # copy baseline file into output dir, so we can use relative paths in html
            # easily package up the results
            baseline_img = "baseline_" + file_name(baseline_img_src)
            shutil.copy(baseline_img_src,pjoin("_output",baseline_img))
            ofile.write('<td><img width="200" src="{0}"><br>baseline</td>\n'.format(baseline_img))
        else:
            ofile.write("<td> BASELINE IMAGE MISSING!</td>\n")
        if "diff_image" in v.keys():
            ofile.write('<td> <img width="200" src="{0}"></td>\n'.format(file_name(v["diff_image"])))
        else:
            ofile.write('<td> (NO DIFF) </td>\n')
        ofile.write("</tr>\n")
    ofile.write("</table>\n")

if __name__ == "__main__" :
    gen_html_report()    
        



