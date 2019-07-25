import sys
# CHANGE this path to the result of:
# $(spack location --install-dir paraview)
paraview_path="/home/danlipsa/projects/ascent_new/build/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/paraview-master-e6ji62w2jt47cj446twsnndkkxgncz6x/lib/python2.7/site-packages"
sys.path.append(paraview_path)


# Same Python interpreter for all time steps
# We use count for one time initializations
try:
    count = count + 1
except NameError:
    count = 0

# ParaView API
# WARNING: this does not work inside the plugin
#          unless you have the same import in paraview-vis.py
from mpi4py import MPI
import paraview
paraview.options.batch = True
paraview.options.symmetric = True
from paraview.simple import *
import ascent_extract

node = ascent_extract.ascent_data().child(0)
domain_id = node["state/domain_id"]
cycle = node["state/cycle"]
imageName = "image_{0:04d}.png".format(int(cycle))
dataName = "paraviewdata_{0:04d}".format(int(cycle))
scriptName = "/home/danlipsa/projects/ascent/src/examples/paraview-vis/paraview_ascent_source.py"
LoadPlugin(scriptName, remote=True, ns=globals())
ascentSource = AscentSource()
ResampleToImage()
rep = Show()

ColorBy(rep, ("POINTS", "nodal_noise"))
# rescale transfer function
nodalNoiseLUT = GetColorTransferFunction('nodal_noise')
nodalNoiseLUT.RescaleTransferFunction(-0.8, 0.8)
# show color bar
renderView1  = GetActiveView()
nodalNoiseLUT = GetScalarBar(nodalNoiseLUT, renderView1)
nodalNoiseLUT.Title = 'nodal_noise'
nodalNoiseLUT.ComponentTitle = ''
# set color bar visibility
nodalNoiseLUT.Visibility = 1
# show color legend
rep.SetScalarBarVisibility(renderView1, True)
rep.SetRepresentationType('Volume')
ResetCamera()
if count == 0:
    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(30)
Render()
WriteImage(imageName)
writer = CreateWriter(dataName + ".pvti", ascentSource)
writer.UpdatePipeline()


# # VTK API
# from ascent_to_vtk import AscentSource, write_vtk
# ascentSource = AscentSource()
# ascentSource.Update()
# write_vtk("vtkdata", ascentSource.GetNode(), ascentSource.GetOutputDataObject(0))


# # Python API
# from ascent_to_vtk import ascent_to_vtk, write_vtk, write_json
# node = ascent_data().child(0)
# write_json("blueprint", node)
# data = ascent_to_vtk(node)
# write_vtk("pythondata", node, data)
