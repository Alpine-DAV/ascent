import sys
# CHANGE this path to the result of:
# $(spack location --install-dir paraview)
paraview_path="/home/danlipsa/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.3.0/paraview-develop-qhm2ssmu2e3ko3jpz6xijn6b64u7nwnp//lib/python2.7/site-packages"
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
if cycle == 200:
    imageName = "moment_{0:04d}.png".format(int(cycle))
    dataName = "cloverleaf3d_data_{0:04d}".format(int(cycle))
    scriptName = "../../paraview-vis/paraview_ascent_source.py"
    LoadPlugin(scriptName, remote=True, ns=globals())
    
    v = OpenDataFile('expandingVortex.vti')
    
    ascentSource = AscentSource()
    
    r = ResampleToImage()
    r.SamplingDimensions = [40, 40, 40]
    
    c = ParallelComputeMoments()
    c.NameOfPointData = "velocity"
    c.Order = 2
    c.NumberOfIntegrationSteps = 0
    c.Radii = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    i=MomentInvariants(InputPattern=v, InputComputedMoments=c)
    i.Order = 2
    i.NameOfPointData = "Result"
    i.NumberOfIntegrationSteps = 0
    i.AngleResolution = 3
    i.IsTranslation = 0
    i.IsScaling = 1
    i.IsRotation = 1
    i.IsReflection = 1
    i.Eps = 0.01
    
    # render the moments
    rep = Show(proxy=i)
    rep.SetRepresentationType('Volume')
    ColorBy(rep, ("POINTS", "0.500000"))
    # rescale transfer function
    tf = GetColorTransferFunction("0.500000")
    tf.RescaleTransferFunction(0.5, 1)
    # show color bar
    renderView1  = GetActiveView()
    scalarBar = GetScalarBar(tf, renderView1)
    scalarBar.Title = '0.500000'
    scalarBar.ComponentTitle = ''
    # set color bar visibility
    scalarBar.Visibility = 1
    # show color legend
    rep.SetScalarBarVisibility(renderView1, True)
    
    
    # # render the data
    # rep = Show()
    # ColorBy(rep, ("CELLS", "energy"))
    # # rescale transfer function
    # energyLUT = GetColorTransferFunction('energy')
    # energyLUT.RescaleTransferFunction(1, 5.5)
    # # show color bar
    # renderView1  = GetActiveView()
    # energyLUT = GetScalarBar(energyLUT, renderView1)
    # energyLUT.Title = 'energy'
    # energyLUT.ComponentTitle = ''
    # # set color bar visibility
    # energyLUT.Visibility = 1
    # # show color legend
    # rep.SetScalarBarVisibility(renderView1, True)
    
    
    ResetCamera()
    if count == 0:
        #cam = GetActiveCamera()
        #cam.Elevation(30)
        #cam.Azimuth(-30)
        pass
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
    
