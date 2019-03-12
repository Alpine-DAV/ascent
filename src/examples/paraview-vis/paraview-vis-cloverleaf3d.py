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
rep = Show()
ColorBy(rep, ("CELLS", "energy"))
# rescale transfer function
energyLUT = GetColorTransferFunction('energy')
energyLUT.RescaleTransferFunction(1, 5.5)
# show color bar
renderView1  = GetActiveView()
energyLUT = GetScalarBar(energyLUT, renderView1)
energyLUT.Title = 'energy'
energyLUT.ComponentTitle = ''
# set color bar visibility
energyLUT.Visibility = 1
# show color legend
rep.SetScalarBarVisibility(renderView1, True)
ResetCamera()
if count == 0:
    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(30)
Render()
# This returns an error:
# X Error of failed request:  BadValue (integer parameter out of range for operation)
#     Major opcode of failed request:  12 (X_ConfigureWindow)
# SaveScreenshot(imageName)
WriteImage(imageName)
writer = CreateWriter(dataName + ".pvtr", ascentSource)
writer.UpdatePipeline()


# # VTK API
# from paraview_ascent_source import AscentSource, write_data
# ascentSource = AscentSource()
# ascentSource.Update()
# write_data("vtkdata", ascentSource.GetJson(), ascentSource.GetOutputDataObject(0))


# # Python API
# from paraview_ascent_source import ascent_to_vtk, write_data, write_json
# json = ascent_data().child(0)
# write_json(json)
# data = ascent_to_vtk(json)
# write_data("pythondata", json, data)
