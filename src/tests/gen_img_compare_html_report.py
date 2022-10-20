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
    # use full path
    return fpath;

def file_name_only(fpath):
    return os.path.basename(fpath)

def find_img_compare_results():
    res = glob.glob(pjoin(output_dir(),"*_img_compare_results.json"))
    res.sort()
    return res

def gen_image_table_entry(fname):
    res  = '<td>\n'
    res += '  <a href="{0}">\n'.format(fname)
    res += '  <img width="200" src="{0}">\n'.format(fname)
    res += '</td>\n'
    return res

def gen_html_report():
    ofile = open(pjoin("_output","tout_img_report.html"),"w")
    ofile.write("<table border=1>\n")
    for res in find_img_compare_results():
        v = json.load(open(res))
        ofile.write("<tr>\n")
        ofile.write("<td>\n")
        case_name = os.path.split(v["test_file"]["path"])[1]
        ofile.write("case: {0} <br>\n".format(case_name))
        if "pass" in v.keys():
            ofile.write("pass: {0} <br>\n".format(v["pass"]))
        for k in ["dims_match","percent_diff","tolerance"]:
            if k in v.keys():
                ofile.write("{0} = {1} <br>\n".format(k,v[k]))
            else:
                ofile.write("{0} = MISSING <br>\n".format(k))
        ofile.write("</td>\n")
        if v["test_file"]["exists"] == "true":
            # copy all things we want in the report to a standard place
            src_file = file_name(v["test_file"]["path"])
            dest_dir = pjoin("_output/_result_images")
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            dest_file = pjoin(dest_dir,file_name_only(src_file))
            shutil.copy(src_file,dest_file)
            ofile.write(gen_image_table_entry(pjoin("_result_images",file_name_only(src_file))))
        else:
            ofile.write("<td> TEST IMAGE MISSING!</td>\n")
        if v["baseline_file"]["exists"] == "true":
            # copy all things we want in the report to a standard place
            src_file = file_name(v["baseline_file"]["path"])
            dest_dir = pjoin("_output/_baseline_images")
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            dest_file = pjoin(dest_dir,file_name_only(src_file))
            shutil.copy(src_file,dest_file)
            ofile.write(gen_image_table_entry(pjoin("_baseline_images",file_name_only(src_file))))
        else:
            ofile.write("<td> BASELINE IMAGE MISSING!</td>\n")
        if "diff_image" in v.keys():
            ofile.write(gen_image_table_entry(file_name(v["diff_image"])))
        else:
            ofile.write('<td> (NO DIFF) </td>\n')
        ofile.write("</tr>\n")
    ofile.write("</table>\n")

if __name__ == "__main__" :
    gen_html_report()
