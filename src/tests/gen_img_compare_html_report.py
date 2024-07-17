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

def test_resources_dir():
    # these are found relative to this script
    return pjoin(os.path.split(os.path.abspath(__file__))[0],"_test_resources")

def gen_html_resources():
    res = glob.glob(pjoin(test_resources_dir(),"*.js"))
    res.extend(glob.glob(pjoin(test_resources_dir(),"*.css")))
    for fname in res:
        shutil.copy2(fname,"_output")

def gen_report_header():
    return """
    <link href="bootstrap.min.css" rel="stylesheet">
    <script src="bootstrap.bundle.min.js"></script>
    <script src="sortable.js"></script>
    <table class="sortable table table-striped table-bordered table-hover">
      <tr>
        <th>Case</th>
        <th>Status</th>
        <th>Details</th>
        <th>Current</th>
        <th>Baseline</th>
        <th>Diff</th>
      </tr>
    <tr>
    """

def gen_html_report():
    gen_html_resources()
    ofile = open(pjoin("_output","index.html"),"w")
    ofile.write(gen_report_header())
    for res in find_img_compare_results():
        v = json.load(open(res))
        ofile.write("<tr>\n")
        # case name
        ofile.write("<td>\n")
        case_name = os.path.split(v["test_file"]["path"])[1]
        ofile.write(case_name)
        ofile.write("</td>\n")
        # status
        ofile.write("<td>\n")
        if "pass" in v.keys():
            ofile.write(v["pass"])
        else:
            ofile.write("unknown")
        ofile.write("</td>\n")
        # details
        # put all info in details
        ofile.write("<td>\n")
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
            ofile.write(gen_image_table_entry(file_name_only(v["diff_image"])))
        else:
            ofile.write('<td> (NO DIFF) </td>\n')
        ofile.write("</tr>\n")
    ofile.write("</table>\n")

if __name__ == "__main__" :
    gen_html_report()
