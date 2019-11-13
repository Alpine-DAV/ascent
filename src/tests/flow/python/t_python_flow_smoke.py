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
 file: t_python_flow_smoke.py
 description: Simple unit test for the basic flow python module interface.

"""

import sys
import unittest

import conduit
import flow


def fake():
    return 1

class SrcFilter(flow.Filter):
    def __init__(self):
        super(SrcFilter, self).__init__()

    def execute(self):
        val = self.params().fetch("value").value()
        self.set_output(val)

    def declare_interface(self,i):
        i.fetch("type_name").set("src");
        i.fetch("output_port").set("true");
        i.fetch("default_params/value").set(0);

class IncFilter(flow.Filter):
    def __init__(self):
        super(IncFilter, self).__init__()

    def execute(self):
        a = self.input("in")
        res = a + 1
        self.set_output(res)

    def declare_interface(self,i):
        i.fetch("type_name").set("inc");
        i.fetch("output_port").set("true");
        port_names = i.fetch("port_names")
        port_names.append().set("in");


class AddFilter(flow.Filter):
    def __init__(self):
        super(AddFilter, self).__init__()

    def execute(self):
        a = self.input("a")
        b = self.input("b")
        res = a + b
        self.set_output(res)

    def declare_interface(self,i):
        i.fetch("type_name").set("add");
        i.fetch("output_port").set("true");
        i.fetch("port_names").append().set("a")
        i.fetch("port_names").append().set("b")


class PrintFilter(flow.Filter):
    def __init__(self):
        super(PrintFilter, self).__init__()

    def execute(self):
        print(self.input("in"))

    def declare_interface(self,i):
        i.fetch("type_name").set("print");
        i.fetch("output_port").set("false");
        i.fetch("port_names").append().set("in")


#
# For an easier way to define a filter, we can wrap a function.
#

def src():
    return 1

def inc(i):
    return i +1

def add(a,b):
    return a+b

def prnt(i):
    print(i)
    return None

class Test_Flow_Basic(unittest.TestCase):
    def test_basic(self):
        flow.Workspace.clear_supported_filter_types()
        flow.Workspace.register_filter_type(SrcFilter)
        flow.Workspace.register_filter_type(IncFilter)
        flow.Workspace.register_filter_type(AddFilter)
        flow.Workspace.register_filter_type(PrintFilter)
        w = flow.Workspace()

        w.graph().add_filter("src","s");
        #
        w.graph().add_filter("inc","a");
        w.graph().add_filter("inc","b");
        w.graph().add_filter("inc","c");

        w.graph().add_filter("print","p");

        #
        w.graph().connect("s","a","in");
        w.graph().connect("a","b","in");
        w.graph().connect("b","c","in");

        w.graph().connect("c","p","in");

        print(w)
        w.execute();

    def test_basic_easy(self):
        flow.Workspace.clear_supported_filter_types()
        flow.Workspace.register_filter_type(flow.wrap_function(src))
        flow.Workspace.register_filter_type(flow.wrap_function(inc))
        flow.Workspace.register_filter_type(flow.wrap_function(add))
        flow.Workspace.register_filter_type(flow.wrap_function(prnt))
        w = flow.Workspace()

        print(w.registry())
        # add our filters
        w.graph().add_filter("src","s");

        w.graph().add_filter("inc","a");
        w.graph().add_filter("inc","b");
        w.graph().add_filter("inc","c");


        # connect everything up
        w.graph().connect("s","a","i");
        w.graph().connect("a","b","i");
        w.graph().connect("b","c","i");

        print(w)
        w.execute();
        print("Reg from Py")
        print(w.registry())

        v = w.registry().fetch("c")

        self.assertEqual(v,4)



if __name__ == '__main__':
    unittest.main()


