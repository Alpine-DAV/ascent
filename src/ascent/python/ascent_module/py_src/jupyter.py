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
import conduit

try:
    # we might not always have jupyter widgets
    # allow import to work
    import ipywidgets
    ipywidgets_support = True
except:
    ipywidgets_support = False

try:
    # cinema jupyter widget
    import cinemasci.pynb
    cinemasci_support = True
except:
    cinemasci_support = False


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
        self.label.value = self.image_path_txt()

    def update_checkbox(self,change):
            self.label.value = self.image_path_txt()

    def image_path_txt(self):
        fname = self.image_fnames[self.image_index]
        if self.check.value is False:
            return "File: " + fname
        else:
            return "File: " + os.path.abspath(fname)
            
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
            v = ipywidgets.VBox([self.title,
                                 self.image,
                                 self.check,
                                 self.label])
        # setup connections the check box (always in use)
        self.check.observe(self.update_checkbox)
        return v

#
# We can't embed the cinema viewer in our own widgets
# I think this is b/c it directly uses IPython.display()
# Skip until we solve this mystery.
#
# class AscentCinemaDBViewer(object):
#     """
#     Widget that shows all created cinema dbs.
#     """
#     def __init__(self, cinema_dbs):
#         self.cinema_dbs = cinema_dbs
#         self.index = 0
#         self.label = ipywidgets.Label(value=self.cinema_dbs[self.index])
#         self.cviewer = cinemasci.pynb.CinemaViewer()
#         self.cviewer.load(self.cinema_dbs[self.index])
#         self.check = ipywidgets.Checkbox(value=False,
#                                       description='Show absolute file path',
#                                       disabled=False,
#                                       indent=False)
#         self.slider = ipywidgets.IntSlider()
#         self.play   = ipywidgets.Play(value=0,
#                                  min=0,
#                                  max=len(self.cinema_dbs)-1,
#                                  step=1,
#                                  interval=500)
#
#         ipywidgets.jslink((self.play, "min"), (self.slider, "min"))
#         ipywidgets.jslink((self.play, "max"), (self.slider, "max"))
#         ipywidgets.jslink((self.play, "value"), (self.slider, "value"))
#
#     def update_index(self,change):
#         self.index = change.owner.value
#         self.cviewer.load(self.cinema_dbs[self.index])
#         self.label.value = self.db_path_txt()
#
#     def update_checkbox(self,change):
#             self.label.value = self.image_path_txt()
#
#     def db_path_txt(self):
#         path = self.cinema_dbs[self.index]
#         if self.check.value is False:
#             return "DB Path: " + fname
#         else:
#             return "DB Path: " + os.path.abspath(path)
#
#     def show(self):
#         if len(self.cinema_dbs) > 1:
#             # box that groups our widgets
#             v = ipywidgets.VBox([self.cviewer,
#                                  self.label,
#                                  self.check,
#                                  self.slider,self.play])
#             # setup connections for this case
#             self.slider.observe(self.update_index)
#             self.play.observe(self.update_index)
#         else:
#             # box that groups our widgets
#             v = ipywidgets.VBox([self.cviewer,
#                                  self.check,
#                                  self.label])
#         # setup connections the check box (always in use)
#         self.check.observe(self.update_checkbox)
#         return v


class AscentStatusWidget(object):
    """
    Simple display of info["status"]
    """
    def __init__(self, info):
        # keep so we can print error details on show
        self.info = info
        style = self.detect_style(info)
        status_title = ipywidgets.Button(description=str(info["status/message"]),
                                         disabled=True,
                                         layout=ipywidgets.Layout(width='100%'),
                                         button_style=style)
        self.main = status_title
        
    def show(self):
        return self.main

    def detect_style(self,info):
        msg = info["status/message"]
        if "failed" in msg:
            return 'danger'
        elif  "Ascent::close" in msg:
            return 'warning'
        else:
            return 'success'

class AscentResultsViewer(object):
    """
    Groups widgets that display Ascent results.
    """
    def __init__(self, info):
        status = AscentStatusWidget(info)
        widget_titles  = []
        widgets = []
        # cinema databases
        # (skipping until we have a handle on embedding cinema widgets)
        # if cinema_support and info.has_child("extracts"):
        #     # filter out any cinema extracts
        #     cinema_exts = conduit.Node()
        #     cinema_paths = []
        #     for c in info["extracts"].children():
        #         if c.node()["type"] == "cinema":
        #             cinema_exts.append().set(c.node())
        #             cinema_paths.append(c.node()["path"])
        #     if cinema_exts.number_of_children() > 0:
        #         w = AscentNodeViewer(cinema_exts)
        #         widget_titles.append("Cinema Databases")
        #         widgets.append(w.show())
        #         w = AscentCinemaDBViewer(cinema_paths)
        #         widget_titles.append("Cinema Databases")
        #         widgets.append(w.show())
        # rendered images
        if info.has_child("images"):
            # get fnames from images section of info
            renders_fnames = []
            for c in info["images"].children():
                renders_fnames.append(c.node()["image_name"])
            w = AscentImageSequenceViewer(renders_fnames)
            widget_titles.append("Renders")
            widgets.append(w.show())
        # extracts
        if info.has_child("extracts"):
            w = AscentNodeViewer(info["extracts"])
            widget_titles.append("Extracts")
            widgets.append(w.show())
        # actions
        if info.has_child("actions"):
            w = AscentNodeViewer(info["actions"])
            widget_titles.append("Actions")
            widgets.append(w.show())
        if len(widgets) > 1:
            # use tabs!
            tab = ipywidgets.Tab()
            tab.children = widgets
            for i,v in enumerate(widget_titles):
                tab.set_title(i,v)
            self.main = ipywidgets.VBox([tab,
                                         status.show()],
                                         layout=ipywidgets.Layout(overflow_x="hidden"))
        elif len(widgets) == 1:
            title = ipywidgets.Label(value=widget_titles[0])
            self.main = ipywidgets.VBox( [title,
                                          widgets[0],
                                          status.show()],
                                         layout=ipywidgets.Layout(overflow_x="hidden"))
        else:
            self.main = ipywidgets.VBox([status.show()],
                                        layout=ipywidgets.Layout(overflow_x="hidden"))


    def show(self):
        return self.main

class AscentNodeViewer(object):
    """
    Shows Yaml Repr of a Conduit Node
    """
    def __init__(self, node):
        v = "<pre>" + node.to_yaml() + "</pre>"
        self.main = ipywidgets.HTML(value=v,
                                    placeholder='',
                                    description='')
    def show(self):
        return self.main

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
        if not self.last_info.has_child("status"):
            raise Exception("ERROR: AscentViewer.refresh_info() failed: info lacks `status`")
        msg = self.last_info["status/message"]
        if "failed" in msg and self.last_info.has_path("status/details"):
            # print to standard notebook output, this looks
            # pretty sensible vs a widget
            print("[Ascent Error]")
            print(self.last_info["status/details"])

    def show(self):
        self.refresh_info()
        if ipywidgets_support:
            self.results_view = AscentResultsViewer(self.last_info)
            return self.results_view.show()
        else:
            raise Exception("ERROR: AscentViewer.show() failed: ipywidgets missing")
