###############################################################################
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
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

###############################################################################
# file: jupyter.py
# Purpose: Jupyter AscentViewer Widget
#
###############################################################################

import os
import ipywidgets
import conduit


class AscentImageSequenceViewer(object):
    """
    Widget that shows all rendered images.
    """
    def __init__(self, image_fnames, title = 'Renders'):
        self.image_display_w = 400
        self.image_display_h = 400
        self.image_fnames = image_fnames
        self.image_index = 0
        self.image_data = []
        for fname in self.image_fnames:
            with open(fname, "rb") as f:
                self.image_data.append(f.read())
        self.title = ipywidgets.Label(value=title)
        self.label = ipywidgets.Label(value=self.image_fnames[self.image_index])
        self.image = ipywidgets.Image(value=self.image_data[self.image_index],
                                      width=self.image_display_w,
                                      height=self.image_display_h,
                                      format="png")
        self.check = ipywidgets.Checkbox(value=False,
                                      description='Show absolute file path',
                                      disabled=False,
                                      indent=False)
        self.slider = ipywidgets.IntSlider()
        self.play   = ipywidgets.Play(value=0,
                                 min=0,
                                 max=len(self.image_data)-1,
                                 step=1,
                                 interval=500) 

        ipywidgets.jslink((self.play, "min"), (self.slider, "min"))
        ipywidgets.jslink((self.play, "max"), (self.slider, "max"))
        ipywidgets.jslink((self.play, "value"), (self.slider, "value"))

    def update_index(self,change):
        self.image_index = change.owner.value
        self.image.value = self.image_data[self.image_index]
        if self.check.value is False:
            self.label.value = "File: " + self.image_fnames[self.image_index]
        else:
            self.label.value = "File: " + os.path.abspath(self.image_fnames[self.image_index])

    def update_checkbox(self,change):
        if self.check.value is False:
            self.label.value = "File: " + self.image_fnames[self.image_index]
        else:
            self.label.value = "File: " + os.path.abspath(self.image_fnames[self.image_index])

    def show(self):
        if len(self.image_data) > 1:
            # box that groups our widgets
            v = ipywidgets.VBox([self.title,
                                 self.image,
                                 self.label,
                                 self.check,
                                 self.slider,self.play])
            # setup connections for this case
            self.slider.observe(self.update_index)
            self.play.observe(self.update_index)
        else:
            # box that groups our widgets
            v = ipywidgets.VBox([self.title,self.image, self.check, self.label])
        # setup connections the check box (always in use)
        self.check.observe(self.update_checkbox)
        return v


class AscentResultsViewer(object):
    """
    Groups widgets that display Ascent results.
    """
    def __init__(self, info):
        renders_fnames  = []
        # get fnames from images section of info
        if info.has_child("images"):
            for c in info["images"].children():
                renders_fnames.append(c.node()["image_name"])
        self.renders = AscentImageSequenceViewer(renders_fnames)
    def show(self):
        return self.renders.show()


class AscentActionsViewer(object):
    """
    Shows Actions from Ascent info.
    """
    def __init__(self, info):
        val = ""
        if info.has_child("actions"):
            val = info["actions"].to_yaml()
        self.title = ipywidgets.Label(value='Actions')
        self.txt = ipywidgets.Textarea(value=val,
                                       placeholder='',
                                       description='',
                                       disabled=False,
                                       layout={'height': '100%'})
        self.box = ipywidgets.VBox([self.title, self.txt],
                                    layout={'height': '500px'})

    def show(self):
        return self.box

class AscentViewer(object):
    """
    A simple jupyter widget ui for Ascent results.
    Wraps an ascent python instance.
     (serves as a replacement for calling the ascent object methods directly)

    Thanks to Tom Stitt @ LLNL who provided a great starting
    point with his image sequence viewer widget. 
    """
    def __init__(self,ascent):
        self.ascent = ascent
        self.show_actions = False

    def open(self, opts=None):
        if not opts is None:
            self.ascent.open(opts)
        else:
            self.ascent.open()

    def info(self):
        n = conduit.Node()
        self.ascent.info(res)
        return res

    def publish(self,data):
        self.ascent.publish(data)

    def execute(self,actions):
        self.ascent.execute(actions)

    def close(self):
        self.ascent.close()

    def refresh_info(self):
        self.last_info  = conduit.Node()
        self.ascent.info(self.last_info)
        
    def toggle_actions(self):
        self.show_actions = not self.show_actions
        self.show()
 
    def show(self):
        self.refresh_info()
        self.results_view = AscentResultsViewer(self.last_info)
        if self.show_actions:
            self.actions_view = AscentActionsViewer(self.last_info)
            self.results_view = AscentResultsViewer(self.last_info)
            self.box = ipywidgets.HBox([self.actions_view.show(),
                                        self.results_view.show()])
            return self.box
        else:
            self.results_view = AscentResultsViewer(self.last_info)
            return self.results_view.show()
