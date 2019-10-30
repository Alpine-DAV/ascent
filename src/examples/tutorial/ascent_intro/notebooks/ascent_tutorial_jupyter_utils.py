import ipywidgets as widgets


class ImageSeqViewer(object):
    def __init__(self,fnames):
        self.data = []
        for fname in fnames:
            with open(fname, "rb") as f:
                self.data.append(f.read())
        self. image = widgets.Image(value=self.data[0],
                                    width=500,
                                    height=500,
                                    format="png")
        self.play = widgets.Play(value=0,
                                 min=0,
                                 max=len(self.data)-1,
                                 step=1,
                                 interval=500) 
    def update(self,change):
        index = change.owner.value
        image.value = self.data[index]
 
    def run(self):
        v = widgets.VBox([self.image, self.play])
        self.play.observe(self.update)
        return v

 