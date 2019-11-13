# Same Python interpreter for all time steps
# We use count for one time initializations
try:
    count = count + 1
except NameError:
    count = 0

if count == 0:
    import sys
    # CHANGE this path to the result of:
    # echo $(spack location --install-dir paraview)/lib/python*/site-packages
    paraview_path="/home/danlipsa/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/paraview-develop-ht7egbxaoqlr2rjg5fju3ic7fej7g47t/lib/python3.7/site-packages"
    sys.path.append(paraview_path)

    # ParaView API
    # WARNING: this does not work inside the plugin
    #          unless you have the same import in paraview-vis.py
    from mpi4py import MPI
    import paraview
    paraview.options.batch = True
    paraview.options.symmetric = True
    from paraview.simple import *
    import ascent_extract

    # CHANGE this path to the result of:
    # echo $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py
    scriptName = "/home/danlipsa/projects/ascent/src/examples/paraview-vis/paraview_ascent_source.py"
    LoadPlugin(scriptName, remote=True, ns=globals())
    ascentSource = AscentSource()
    ResampleToImage()
    rep = Show()

    ColorBy(rep, ("POINTS", "phi"))
    # rescale transfer function
    phiLUT = GetColorTransferFunction('phi')
    phiLUT.RescaleTransferFunction(-0.14, 2)
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

    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(30)

node = ascent_extract.ascent_data().child(0)
domain_id = node["state/domain_id"]
cycle = node["state/cycle"]
imageName = "image_{0:04d}.png".format(int(cycle))
dataName = "paraviewdata_{0:04d}".format(int(cycle))
ascentSource.Count = count
ResetCamera()
Render()
SaveScreenshot(imageName, ImageResolution=(1024, 1024))
# writer = CreateWriter(dataName + ".pvtr", ascentSource)
# writer.UpdatePipeline()


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
