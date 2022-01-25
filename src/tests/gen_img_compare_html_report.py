###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
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
        



