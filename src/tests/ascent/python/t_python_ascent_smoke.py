###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: t_python_ascent_smoke.py
 description: Simple unit test for the basic ascent python module interface.

"""

import sys
import unittest

import conduit
import ascent

class Test_Ascent_Basic(unittest.TestCase):
    def test_about(self):
        print(ascent.about())
        s = ascent.Ascent()

if __name__ == '__main__':
    unittest.main()


