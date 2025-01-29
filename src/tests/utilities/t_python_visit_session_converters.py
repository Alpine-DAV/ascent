###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: t_python_visit_session_converts.py
 description: Driver to test python session converters

"""

import sys
import unittest
import os
import subprocess

from os.path import join as pjoin


def test_src_dir():
    for path in sys.path:
        if os.path.isfile(pjoin(path,"tin-visit-cam.session")):
            return path

def utils_src_dir():
    res = os.path.abspath(pjoin(test_src_dir(),"..","..","utilities"))
    print(res)
    return res 

class Test_Session_Converters(unittest.TestCase):

    def test_extract_camera(self):
        test_script = pjoin(utils_src_dir(),"visit_session_converters","session_to_camera.py")
        test_sess   = pjoin(test_src_dir(),"tin-visit-cam.session")
        print(test_script)
        print(test_sess)
        cmd = " ".join([sys.executable,test_script,test_sess])
        print(cmd)
        subprocess.check_call(cmd,shell=True)

    def test_extract_opac(self):
        test_script = pjoin(utils_src_dir(),"visit_session_converters","session_to_opacity.py")
        test_sess   = pjoin(test_src_dir(),"tin-visit-opac.session")
        print(test_script)
        print(test_sess)
        cmd = " ".join([sys.executable,test_script,test_sess])
        print(cmd)
        subprocess.check_call(cmd,shell=True)

    def test_extract_ct(self):
        test_script = pjoin(utils_src_dir(),"visit_session_converters","visit_color_table_to_ascent_color_table.py")
        test_sess   = pjoin(test_src_dir(),"batlow-visit-ct.ct")
        print(test_script)
        print(test_sess)
        cmd = " ".join([sys.executable,test_script,test_sess])
        print(cmd)
        subprocess.check_call(cmd,shell=True)

if __name__ == '__main__':
    unittest.main()


