import ipywidgets as widgets

# common size for displaying rendered results
img_display_width = 400

# helper for displaying sequences of images
# in a jupyter notebook (thanks to Tom Stitt @ LLNL)
class ImageSeqViewer(object):
    def __init__(self,fnames):
        self.data = []
        for fname in fnames:
            with open(fname, "rb") as f:
                self.data.append(f.read())
        self.image = widgets.Image(value=self.data[0],
                                    width=img_display_width,
                                    height=img_display_width,
                                    format="png")
        self.slider = widgets.IntSlider()
        self.play = widgets.Play(value=0,
                                 min=0,
                                 max=len(self.data)-1,
                                 step=1,
                                 interval=500) 

        widgets.jslink((self.play, "min"), (self.slider, "min"))
        widgets.jslink((self.play, "max"), (self.slider, "max"))
        widgets.jslink((self.play, "value"), (self.slider, "value"))

    def update(self,change):
        index = change.owner.value
        self.image.value = self.data[index]
 
    def show(self):
        v = widgets.VBox([self.image, self.slider, self.play])
        self.slider.observe(self.update)
        self.play.observe(self.update)
        return v
