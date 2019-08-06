import sys
from pathlib import Path
home = str(Path.home())
# CHANGE this path to the result of:
# $(spack location --install-dir paraview)
paraview_path=home + "/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.3.0/paraview-develop-qhm2ssmu2e3ko3jpz6xijn6b64u7nwnp//lib/python2.7/site-packages"
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
from paraview.simple import LoadPlugin, Show, ColorBy,\
    GetColorTransferFunction, GetActiveView, ResetCamera, GetActiveCamera,\
    GetScalarBar, Render, WriteImage, CreateWriter
import ascent_extract

node = ascent_extract.ascent_data().child(0)
domain_id = node["state/domain_id"]
cycle = node["state/cycle"]
imageName = "image_{0:04d}.png".format(int(cycle))
dataName = "paraviewdata_{0:04d}".format(int(cycle))
# CHANGE this path to the result of:
# echo $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py
scriptName = home + "/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.3.0/ascent-develop-ncrcrl6ib5k6zkcffd6k6repvkk3b44e/examples/ascent/paraview-vis/paraview_ascent_source.py"
LoadPlugin(scriptName, remote=True, ns=globals())
ascentSource = AscentSource()
# show the boundary topology. For the main topology use Port 0.
ascentSource.Port = 1
rep = Show()

ColorBy(rep, ("CELLS", "boundary_attribute"))
# rescale transfer function
lut = GetColorTransferFunction('boundary_attribute')
lut.RescaleTransferFunction(1, 3)
# show color bar
renderView1 = GetActiveView()
lut = GetScalarBar(lut, renderView1)
lut.Title = 'boundary_attribute'
lut.ComponentTitle = ''
# set color bar visibility
lut.Visibility = 1
# show color legend
rep.SetScalarBarVisibility(renderView1, True)
ResetCamera()
if count == 0:
    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(30)
Render()
WriteImage(imageName)
writer = CreateWriter(dataName + ".pvtu", ascentSource)
writer.UpdatePipeline()


# # VTK API
# from ascent_to_vtk import AscentSource, write_vtk
# ascentSource = AscentSource()
# ascentSource.Update()
# write_vtk("vtkdata-main", ascentSource.GetNode(),
#            ascentSource.GetOutputDataObject(0))


# # Python API
# from ascent_to_vtk import ascent_to_vtk, write_vtk, write_json
# node = ascent_data().child(0)
# write_json("blueprint", node)
# data = ascent_to_vtk(node, "main")
# write_vtk("pythondata-main", node, data)
# data = ascent_to_vtk(node, "boundary")
# write_vtk("pythondata-boundary", node, data)
