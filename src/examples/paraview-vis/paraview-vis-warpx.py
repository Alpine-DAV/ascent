# Same Python interpreter for all time steps
# We use count for one time initializations
try:
    count = count + 1
except NameError:
    count = 0

if count == 0:
    import sys
    # CHANGE this path to the result of:
    # echo $(spack location --install-dir paraview)
    paraview_path="/home/danlipsa/projects/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/paraview-develop-ht7egbxaoqlr2rjg5fju3ic7fej7g47t"
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

    # create the pipeline
    ascentSource = AscentSource()
    asDisplay = Show()
    asDisplay.SetRepresentationType('Outline')

    contour = Contour()
    contour.ContourBy = ["POINTS", "Ex"]
    contour.ComputeScalars = True
    contour.Isosurfaces = [-1000000000, 0, 1000000000]
    cDisplay = Show()

    # r = ResampleToImage()
    # r.SamplingDimensions = [40, 40, 40]

    # # render the moments
    # rDisplay = Show(proxy=r)
    # rDisplay.SetRepresentationType('Volume')

    ColorBy(cDisplay, ("POINTS", "Ex"))
    # rescale transfer function
    ctf = GetColorTransferFunction("Ex")
    ctf.RescaleTransferFunction(-1.4e+9, 1.4e+9)

    # otf = GetOpacityTransferFunction("Ex")
    # # otf.Points = [-1432875392.0, 1.0, 0.0, 0.0, 1432875453.4194217, 1.0]

    # # show color bar
    # renderView1  = GetActiveView()
    # scalarBar = GetScalarBar(ctf, renderView1)
    # scalarBar.Title = 'Ex'
    # scalarBar.ComponentTitle = ''
    # # set color bar visibility
    # scalarBar.Visibility = 1
    # # show color legend
    # rDisplay.SetScalarBarVisibility(renderView1, True)

    cam = GetActiveCamera()
    cam.Elevation(30)
    cam.Azimuth(-30)



node = ascent_extract.ascent_data().child(0)
domain_id = node["state/domain_id"]
cycle = node["state/cycle"]
imageName = "warpx_{0:04d}.png".format(int(cycle))
dataName = "warpx_data_{0:04d}".format(int(cycle))
ascentSource.Count = count
Render()
SaveScreenshot(imageName, ImageResolution=(1024, 1024), FontScaling='Do not scale fonts')
