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
    paraview_path = ('/home/danlipsa/projects/spack/opt/spack/'
                     'linux-ubuntu18.04-x86_64/gcc-7.4.0/'
                     'paraview-develop-odtqjudmv53um5qyzvfojkcbyi64glmc')
    sys.path.append(paraview_path)

    # ParaView API
    # WARNING: this does not work inside the plugin
    #          unless you have the same import in paraview-vis.py
    import paraview
    paraview.options.batch = True
    paraview.options.symmetric = True
    from paraview.simple import LoadPlugin, Contour, \
        Show, ColorBy, GetColorTransferFunction, GetActiveCamera, Render,\
        SaveScreenshot
    import ascent_extract

    # CHANGE this path to the result of:
    # echo $(spack location --install-dir ascent)/
    #                  examples/ascent/paraview-vis/paraview_ascent_source.py
    scriptName = ('/home/danlipsa/projects/ascent/src/examples/'
                  'paraview-vis/paraview_ascent_source.py')
    LoadPlugin(scriptName, remote=True, ns=globals())

    # create the pipeline
    ascentSource = AscentSource()
    asDisplay = Show()
    asDisplay.SetRepresentationType('Outline')

    contour = Contour()
    contour.ContourBy = ['POINTS', 'Ex']
    contour.ComputeScalars = True
    contour.Isosurfaces = [-129000000, 129000000]
    cDisplay = Show()
    cDisplay.Opacity = 0.3

    ColorBy(cDisplay, ('POINTS', 'Ex'))
    # rescale transfer function
    ctf = GetColorTransferFunction('Ex')
    ctf.RescaleTransferFunction(-258766162, 258766162)

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
domain_id = node['state/domain_id']
cycle = node['state/cycle']
imageName = 'warpx_{0:04d}.png'.format(int(cycle))
dataName = 'warpx_data_{0:04d}'.format(int(cycle))
ascentSource.Count = count
Render()
SaveScreenshot(imageName, ImageResolution=(1024, 1024),
               FontScaling='Do not scale fonts')
# writer = CreateWriter(dataName + '.pvtr', ascentSource)
# writer.UpdatePipeline()
