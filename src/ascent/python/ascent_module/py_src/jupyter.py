###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: jupyter.py
# Purpose: Jupyter Ascent Widgets
#
###############################################################################

import os
import conduit

try:
    # we might not always have ipywidgets module (jupyter widgets)
    # allow import to work even if not installed
    import ipywidgets
    ipywidgets_support = True
except:
    ipywidgets_support = False

try:
    # we might not always have the cinemasci module
    # allow import to work even if not installed
    import cinemasci.pynb
    cinemasci_support = True
except:
    cinemasci_support = False

class AscentImageSequenceViewer(object):
    """
    Widget that shows a sequence of images with play controls.
    """
    def __init__(self, image_fnames):
        self.display_w = 400
        self.display_h = 400
        self.fnames = image_fnames
        self.index = 0
        self.data = []
        for fname in self.fnames:
            with open(fname, "rb") as f:
                self.data.append(f.read())
        self.label = ipywidgets.Label(value=self.fnames[self.index])
        self.image = ipywidgets.Image(value=self.data[self.index],
                                      width=self.display_w,
                                      height=self.display_h,
                                      format="png")
        self.check = ipywidgets.Checkbox(value=False,
                                      description='Show absolute file path',
                                      disabled=False,
                                      indent=False)
        self.slider = ipywidgets.IntSlider()
        self.play   = ipywidgets.Play(value=0,
                                 min=0,
                                 max=len(self.data)-1,
                                 step=1,
                                 interval=500)
        ipywidgets.jslink((self.play, "min"), (self.slider, "min"))
        ipywidgets.jslink((self.play, "max"), (self.slider, "max"))
        ipywidgets.jslink((self.play, "value"), (self.slider, "value"))

    def update_index(self,change):
        self.index = change.owner.value
        self.image.value = self.data[self.index]
        self.label.value = self.image_path_txt()

    def update_checkbox(self,change):
        self.label.value = self.image_path_txt()

    def image_path_txt(self):
        fname = self.fnames[self.index]
        if self.check.value is False:
            return "File: " + fname
        else:
            return "File: " + os.path.abspath(fname)

    def show(self):
        if len(self.data) > 1:
            # if we have more than one image
            # we enabled the slider + play controls
            v = ipywidgets.VBox([self.image,
                                 self.label,
                                 self.check,
                                 self.slider,self.play])
            # setup connections for our controls
            self.slider.observe(self.update_index)
            self.play.observe(self.update_index)
        else:
            # if we have one image, we only show
            # the label and abs checkbox
            v = ipywidgets.VBox([self.image,
                                 self.check,
                                 self.label])
        # setup connections the check box (always in use)
        self.check.observe(self.update_checkbox)
        return v

class AscentStatusWidget(object):
    """
    Simple display of info["status"]
    """
    def __init__(self, info):
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

class AscentNodeViewer(object):
    """
    Shows YAML repr of a Conduit Node
    """
    def __init__(self, node):
        v = "<pre>" + node.to_yaml() + "</pre>"
        self.main = ipywidgets.HTML(value=v,
                                    placeholder='',
                                    description='')

    def show(self):
        return self.main


class AscentResultsViewer(object):
    """
    Widget that display Ascent results from info.
    """
    def __init__(self, info):
        status = AscentStatusWidget(info)
        widget_titles  = []
        widgets = []

        # cinema databases
        # (skipping until we have a handle on embedding cinema widgets)
        # if cinema_support and info.has_child("extracts"):
        #     # collect any cinema extracts
        #     cinema_exts = conduit.Node()
        #     cinema_paths = []
        #     for c in info["extracts"].children():
        #         if c.node()["type"] == "cinema":
        #             cinema_exts.append().set(c.node())
        #     if cinema_exts.number_of_children() > 0:
        #         w = AscentNodeViewer(cinema_exts)
        #         widget_titles.append("Cinema Databases")
        #         widgets.append(w.show())

        # rendered images
        if info.has_child("images"):
            # get fnames from images section of info
            renders_fnames = []
            for c in info["images"].children():
                renders_fnames.append(c.node()["image_name"])
            # view using image viewer widget
            w = AscentImageSequenceViewer(renders_fnames)
            widget_titles.append("Renders")
            widgets.append(w.show())

        # extracts
        if info.has_child("extracts"):
            # view results info as yaml repr
            w = AscentNodeViewer(info["extracts"])
            widget_titles.append("Extracts")
            widgets.append(w.show())

        # expressions
        if info.has_child("expressions"):
            # view results info as yaml repr
            w = AscentNodeViewer(info["expressions"])
            widget_titles.append("Expressions")
            widgets.append(w.show())

        # actions
        if info.has_child("actions"):
            # view results info as yaml repr
            w = AscentNodeViewer(info["actions"])
            widget_titles.append("Actions")
            widgets.append(w.show())

        # layout active widgets
        if len(widgets) > 0:
            # multiple widgets, group into tabs
            tab = ipywidgets.Tab()
            tab.children = widgets
            for i,v in enumerate(widget_titles):
                tab.set_title(i,v)
            self.main = ipywidgets.VBox([tab,
                                         status.show()],
                                         layout=ipywidgets.Layout(overflow_x="hidden"))
        else:
            # only the status button
            self.main = ipywidgets.VBox([status.show()],
                                        layout=ipywidgets.Layout(overflow_x="hidden"))

    def show(self):
        return self.main


class AscentViewer(object):
    """
    A jupyter widget ui for Ascent results.

    Thanks to Tom Stitt @ LLNL who provided a great starting
    point with his image sequence viewer widget. 
    """
    def __init__(self,ascent):
        self.ascent = ascent

    def show(self):
        info  = conduit.Node()
        self.ascent.info(info)
        if not info.has_child("status"):
            raise Exception("ERROR: AscentViewer.show() failed: info lacks `status`")
        msg = info["status/message"]
        if "failed" in msg and info.has_path("status/details"):
            # print error details to standard notebook output, 
            # this looks pretty sensible vs trying to format in a widget
            print("[Ascent Error]")
            print(info["status/details"])
        if ipywidgets_support:
            view = AscentResultsViewer(info)
            return view.show()
        else:
            raise Exception("ERROR: AscentViewer.show() failed: ipywidgets missing")
