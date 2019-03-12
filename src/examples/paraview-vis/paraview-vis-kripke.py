import sys
# CHANGE this path to the result of:
# $(spack location --install-dir paraview)
paraview_path="/home/danlipsa/projects/ascent/build/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.3.0/paraview-master-dn6y4ofccm4eg3wya6u5uvgz6x6aar5w"
paraview_path = paraview_path + "/lib/python2.7/site-packages"
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

json = ascent_extract.ascent_data().child(0)
domain_id = json["state/domain_id"]
cycle = json["state/cycle"]
imageName = "image_{0:04d}.png".format(int(cycle))
dataName = "paraviewdata_{0:04d}".format(int(cycle))
scriptName = "/home/danlipsa/projects/ascent/src/examples/paraview-vis/paraview_ascent_source.py"
LoadPlugin(scriptName, remote=True, ns=globals())
ascentSource = AscentSource()
ResampleToImage()
rep = Show()

ColorBy(rep, ("POINTS", "phi"))
# rescale transfer function
phiLUT = GetColorTransferFunction('phi')
phiLUT.RescaleTransferFunction(-0.14, 7.2)
# show color bar
renderView1  = GetActiveView()
phiLUT = GetScalarBar(phiLUT, renderView1)
phiLUT.Title = 'phi'
phiLUT.ComponentTitle = ''
# set color bar visibility
phiLUT.Visibility = 1
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
writer = CreateWriter(dataName + ".pvtr", ascentSource)
writer.UpdatePipeline()


# # VTK API
# from ascent_to_vtk import AscentSource, write_data
# ascentSource = AscentSource()
# ascentSource.Update()
# write_data("vtkdata", ascentSource.GetJson(), ascentSource.GetOutputDataObject(0))


# # Python API
# from ascent_to_vtk import ascent_to_vtk, write_data, write_json
# json = ascent_data().child(0)
# write_json(json)
# data = ascent_to_vtk(json)
# write_data("pythondata", json, data)
