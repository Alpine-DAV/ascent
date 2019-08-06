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
from paraview.simple import *
import ascent_extract


node = ascent_extract.ascent_data().child(0)
domain_id = node["state/domain_id"]
cycle = node["state/cycle"]
imageName = "image_{0:04d}.png".format(int(cycle))
dataName = "cloverleaf3d_data_{0:04d}".format(int(cycle))
# CHANGE this path to the result of:
# echo $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py
scriptName = home + "/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.3.0/ascent-develop-ncrcrl6ib5k6zkcffd6k6repvkk3b44e/examples/ascent/paraview-vis/paraview_ascent_source.py"
LoadPlugin(scriptName, remote=True, ns=globals())

ascentSource = AscentSource()

# ghosts are not supported in rectilinear grids so we just extract the regular cells.
sel=SelectCells("vtkGhostType < 1")
e = ExtractSelection(Selection=sel)
# render the data
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
    cam.Azimuth(-30)
Render()


SaveScreenshot(imageName, ImageResolution=(1024, 1024))
# writer = CreateWriter(dataName + ".pvtr", ascentSource)
# writer.UpdatePipeline()


# # VTK API
# from paraview_ascent_source import AscentSource, write_vtk
# ascentSource = AscentSource()
# ascentSource.Update()
# write_vtk("vtkdata", ascentSource.GetNode(), ascentSource.GetOutputDataObject(0))


# # Python API
# from paraview_ascent_source import ascent_to_vtk, write_vtk, write_json, write_hdf
# node = ascent_data().child(0)
# write_json("blueprint", node)
# write_hdf("data", node)
# data = ascent_to_vtk(node)
# write_vtk("pythondata", node, data)
    
