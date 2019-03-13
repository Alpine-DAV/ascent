import numpy as np
from vtkmodules.util import numpy_support, vtkConstants
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData,\
     vtkRectilinearGrid, vtkStructuredGrid,\
     vtkUnstructuredGrid, vtkCellArray, vtkDataObject
from vtkmodules.vtkCommonExecutionModel \
  import vtkStreamingDemandDrivenPipeline, vtkAlgorithm
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter,\
     vtkXMLRectilinearGridWriter, vtkXMLStructuredGridWriter,\
     vtkXMLUnstructuredGridWriter
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkParallelMPI4Py import vtkMPI4PyCommunicator
from vtkmodules.vtkParallelMPI import vtkMPIController
from vtkmodules.vtkParallelCore import vtkMultiProcessController
# needed for paraview decorators
from paraview.util.vtkAlgorithm import smproxy, smproperty
import ascent_extract

# keep arounds a reference to python arrays passed to VTK so that
# they don't get deleted
_keep_around = []


def read_fields(node, topology, data):
    '''
    Read fields form Ascent 'node' into VTK 'data'
    '''
    for fieldIt in node["fields"].children():
        field = fieldIt.node()
        if (field["topology"] != topology):
            continue
        values = field["values"]
        va = None
        if type(values) is np.ndarray:
            va = numpy_support.numpy_to_vtk(values)
        else:
            # structure of arrays
            components = []
            for component in values:
                components.append(values[component.name()])
            vn = np.stack(components, axis=1)
            _keep_around.append(vn)
            va = numpy_support.numpy_to_vtk(vn)
        va.SetName(fieldIt.name())
        if field["association"] == "element":
            data.GetCellData().AddArray(va)
        elif field["association"] == "vertex":
            data.GetPointData().AddArray(va)
        else:
            # null
            # skip higher order attributes for now
            continue


def topology_names(node):
    '''
    Read children names for 'topologies'
    '''
    parent = "topologies"
    children = []
    for c in node[parent].children():
        children.append(c.name())
    return children


def ascent_to_vtk(node, topology=None, extent=None):
    '''
    Read from Ascent node ("topologies/" + topology) into VTK data.
    topology is one of the names returned by topology_names(node) or
    topology_names(node)[0] if the parameter is None
    '''
    global _keep_around
    # we use the same Python interpreter between time steps
    _keep_around = []
    if topology is None:
        topology = topology_names(node)[0]
    data = None
    coords = node["topologies/" + topology + "/coordset"]
    if (node["topologies/" + topology + "/type"] == "uniform"):
        # tested with noise
        data = vtkImageData()
        origin = np.array([float(o) for o in
                           [node["coordsets/" + coords + "/origin/x"],
                            node["coordsets/" + coords + "/origin/y"],
                            node["coordsets/" + coords + "/origin/z"]]])
        spacing = np.array([float(s) for s in
                            [node["coordsets/" + coords + "/spacing/dx"],
                             node["coordsets/" + coords + "/spacing/dy"],
                             node["coordsets/" + coords + "/spacing/dz"]]])
        if extent is None:
            data.SetDimensions(node["coordsets/" + coords + "/dims/i"],
                               node["coordsets/" + coords + "/dims/j"],
                               node["coordsets/" + coords + "/dims/k"])
            data.SetOrigin(origin)
        else:
            data.SetExtent(extent)
            origin = origin - np.array(
                [extent[0], extent[2], extent[4]]) * spacing
            data.SetOrigin(origin)
        data.SetSpacing(spacing)
    elif (node["topologies/" + topology + "/type"] == "rectilinear"):
        # tested on cloverleaf3d and kripke
        data = vtkRectilinearGrid()
        xn = node["coordsets/" + coords + "/values/x"]
        xa = numpy_support.numpy_to_vtk(xn)
        data.SetXCoordinates(xa)

        yn = node["coordsets/" + coords + "/values/y"]
        ya = numpy_support.numpy_to_vtk(yn)
        data.SetYCoordinates(ya)

        zn = node["coordsets/" + coords + "/values/z"]
        za = numpy_support.numpy_to_vtk(zn)
        data.SetZCoordinates(za)
        if (extent is None):
            data.SetDimensions(xa.GetNumberOfTuples(),
                               ya.GetNumberOfTuples(),
                               za.GetNumberOfTuples())
        else:
            data.SetExtent(extent)
    elif (node["coordsets/" + coords + "/type"] == "explicit"):
        xn = node["coordsets/" + coords + "/values/x"]
        yn = node["coordsets/" + coords + "/values/y"]
        zn = node["coordsets/" + coords + "/values/z"]
        xyzn = np.stack((xn, yn, zn), axis=1)
        _keep_around.append(xyzn)
        xyza = numpy_support.numpy_to_vtk(xyzn)
        points = vtkPoints()
        points.SetData(xyza)
        if (node["topologies/" + topology + "/type"] == "structured"):
            # tested on lulesh
            data = vtkStructuredGrid()
            data.SetPoints(points)
            # elements are one less than points
            nx = node["topologies/" + topology + "/elements/dims/i"] + 1
            ny = node["topologies/" + topology + "/elements/dims/j"] + 1
            nz = node["topologies/" + topology + "/elements/dims/k"] + 1
            data.SetDimensions(nx, ny, nz)
        elif (node["topologies/" + topology + "/type"] == "unstructured"):
            # tested with laghos
            data = vtkUnstructuredGrid()
            data.SetPoints(points)
            shape = node["topologies/" + topology + "/elements/shape"]
            c = node["topologies/" + topology +
                     "/elements/connectivity"]
            # vtkIdType is int64
            c = c.astype(np.int64)
            if (shape == "hex"):
                npoints = 8
                cellType = vtkConstants.VTK_HEXAHEDRON
            elif (shape == "quad"):
                npoints = 4
                cellType = vtkConstants.VTK_QUAD
            else:
                print("Error: Shape not implemented")
                return None
            c = c.reshape(c.shape[0] / npoints, npoints)
            # insert the number of points before point references
            c = np.insert(c, 0, npoints, axis=1)
            ncells = c.shape[0]
            c = c.flatten()
            _keep_around.append(c)
            ita = numpy_support.numpy_to_vtkIdTypeArray(c)
            cells = vtkCellArray()
            cells.SetCells(ncells, ita)
            data.SetCells((cellType), cells)
    read_fields(node, topology, data)
    return data


def write_json(node, prefix="blueprint"):
    '''
    Write the Ascent node object into a json file
    '''
    # MPI domain
    domain_id = node["state/domain_id"]
    # time step
    cycle = node["state/cycle"]
    fileNoExt = (prefix + "_" + str(domain_id) +
                 "_{0:04d}").format(int(cycle))
    with open(fileNoExt + ".json", "w") as f:
        f.write("{}".format(node))


def write_data(prefix, node, data):
    '''
    Write the VTK data (read from an Ascent node object) to disk.
    Note that data is not written to a parallel file, each node writes its
    own data.
    '''
    writer = None
    extension = ""
    if data is None:
        print("Error: None data.")
        return
    if data.GetClassName() == "vtkRectilinearGrid":
        writer = vtkXMLRectilinearGridWriter()
        extension = "vtr"
    elif data.GetClassName() == "vtkStructuredGrid":
        writer = vtkXMLStructuredGridWriter()
        extension = "vts"
    elif data.GetClassName() == "vtkImageData":
        writer = vtkXMLImageDataWriter()
        extension = "vti"
    elif data.GetClassName() == "vtkUnstructuredGrid":
        writer = vtkXMLUnstructuredGridWriter()
        extension = "vtu"
    else:
        print("Error: Unknown datatype.")
        return
    # MPI domain
    domain_id = node["state/domain_id"]
    # time step
    cycle = node["state/cycle"]
    fileNoExt = (prefix + "_" + str(domain_id) +
                 "_{0:04d}").format(int(cycle))
    writer.SetFileName(fileNoExt + "." + extension)
    writer.SetInputDataObject(data)
    writer.Write()


def extents(originDims, comm, mpi_size, mpi_rank, topology):
    '''
    Computes whole_extent and extent for rectilinear and uniform topologies.
    originDims: start coordinate and number of values for each of x, y, z
    '''
    whole_extent = np.zeros(6, dtype=int)
    extent = np.zeros(6, dtype=int)
    a = np.zeros(len(originDims) * mpi_size, dtype=np.float32)
    comm.Allgather(originDims, a)
    a = a.reshape(mpi_size, len(originDims))
    for i in range(3):
        coord = a[:, 2*i]
        size = a[:, 2*i+1].astype(int)
        rank = np.arange(mpi_size)
        if np.all(coord == coord[0]):
            # all coordinates are equal
            extent[2*i] = 0
            extent[2*i + 1] = size[i] - 1
            whole_extent[2*i] = extent[2*i]
            whole_extent[2*i+1] = extent[2*i+1]
        else:
            indexSort = np.argsort(coord)
            coord = coord[indexSort]
            size = size[indexSort]
            rank = rank[indexSort]
            # mark with True jumps in coord value:
            # jumps[0] shows the jump at coord[1]
            jumps = (np.ediff1d(coord) != 0)
            jumps = np.insert(jumps, 0, False)
            # store size before the jump at the jump index
            startIndex = np.where(jumps,
                                  np.insert(size[1:], 0, 0),
                                  np.zeros(mpi_size, dtype=int))
            startIndex = np.cumsum(startIndex)
            # remove overlapping points
            overlap = np.cumsum(jumps)
            startIndex = startIndex - overlap
            # compute extents
            whole_extent[2*i] = startIndex[0]
            whole_extent[2*i+1] = startIndex[-1] + size[rank[-1]] - 1
            rankIndex = np.flatnonzero(rank == mpi_rank)[0]
            extent[2*i] = startIndex[rankIndex]
            extent[2*i+1] = startIndex[rankIndex] + size[mpi_rank] - 1
    return (whole_extent, extent)


def extents_rectilinear(node, comm, mpi_size, mpi_rank, topology):
    '''
    Computes whole_extent and extent for the current node for a rectilinear
    dataset described in node.
    '''
    coords = node["topologies/" + topology + "/coordset"]
    xn = node["coordsets/" + coords + "/values/x"]
    yn = node["coordsets/" + coords + "/values/y"]
    zn = node["coordsets/" + coords + "/values/z"]
    originDims = np.array([xn[0], len(xn), yn[0], len(yn), zn[0], len(zn)],
                          dtype=np.float32)
    return extents(originDims, comm, mpi_size, mpi_rank, topology)


def extents_uniform(node, comm, mpi_size, mpi_rank, topology):
    '''
    Computes whole_extent and extent for the current node for an uniform
    dataset described in node.
    '''
    coords = node["topologies/" + topology + "/coordset"]
    o = [node["coordsets/" + coords + "/origin/x"],
         node["coordsets/" + coords + "/origin/y"],
         node["coordsets/" + coords + "/origin/z"]]
    dims = [node["coordsets/" + coords + "/dims/i"],
            node["coordsets/" + coords + "/dims/j"],
            node["coordsets/" + coords + "/dims/k"]]
    originDims = np.array([o[0], dims[0], o[1], dims[1], o[2], dims[2]],
                          dtype=np.float32)
    return extents(originDims, comm, mpi_size, mpi_rank, topology)


@smproxy.source(name="AscentSource",
                label="AscentSource")
class AscentSource(VTKPythonAlgorithmBase):
    """
    Reads data from Ascent insitu library using the blueprint format.
    Multiple topologies are output on multiple output ports.
    """
    def __init__(self):
        # WARNING: this does not work inside the plugin
        #          unless you have the same import in paraview-vis.py
        from mpi4py import MPI
        self._node = ascent_extract.ascent_data().child(0)
        # topology and coords are set only if there is only one topology,
        # otherwise they are none.
        self._topology = None
        self._coords = None
        self._outputType = None
        self._extension = None
        self._comm = MPI.Comm.f2py(ascent_extract.ascent_mpi_comm_id())
        self._mpi_rank = self._comm.Get_rank()
        self._mpi_size = self._comm.Get_size()
        self._whole_extent = np.zeros(6, dtype=int)
        self._extent = np.zeros(6, dtype=int)
        self._topologies = topology_names(self._node)
        # setup VTK / ParaView global controler
        vtkComm = vtkMPI4PyCommunicator.ConvertToVTK(self._comm)
        controller = vtkMultiProcessController.GetGlobalController()
        if controller is None:
            controller = vtkMPIController()
            vtkMultiProcessController.SetGlobalController(controller)
        controller.SetCommunicator(vtkComm)
        # we have as many output ports as topologies
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=len(self._topologies),
            outputType=None)

    def GetNode(self):
        return self._node

    def RequestData(self, request, inVector, outVector):
        for outputPort in range(self.GetNumberOfOutputPorts()):
            oi = outVector.GetInformationObject(outputPort)
            if (oi.Has(vtkStreamingDemandDrivenPipeline.UPDATE_PIECE_NUMBER())
                and oi.Has(vtkStreamingDemandDrivenPipeline.
                           UPDATE_NUMBER_OF_PIECES())):
                requestedPiece = oi.Get(
                    vtkStreamingDemandDrivenPipeline.UPDATE_PIECE_NUMBER())
                numberOfPieces = oi.Get(vtkStreamingDemandDrivenPipeline.
                                        UPDATE_NUMBER_OF_PIECES())
                if requestedPiece != self._mpi_rank or\
                   numberOfPieces != self._mpi_size:
                    print("Warning: Process numbers do not match "
                          "piece numbers {} != {}".format(self._mpi_rank,
                                                          requestedPiece))
            output = oi.Get(vtkDataObject.DATA_OBJECT())
            data = ascent_to_vtk(self._node, self._topologies[outputPort],
                                 self._extent)
            output.ShallowCopy(data)
        return 1

    def RequestInformation(self, request, inVector, outVector):
        for outputPort in range(self.GetNumberOfOutputPorts()):
            oi = outVector.GetInformationObject(outputPort)
            if (self._outputType != vtkConstants.VTK_UNSTRUCTURED_GRID):
                dims = (0, 0, 0)
                if (self._outputType == vtkConstants.VTK_IMAGE_DATA):
                    dims = (
                        self._node["coordsets/" + self._coords + "/dims/i"],
                        self._node["coordsets/" + self._coords + "/dims/j"],
                        self._node["coordsets/" + self._coords + "/dims/k"])
                    origin = (
                        self._node["coordsets/" + self._coords + "/origin/x"],
                        self._node["coordsets/" + self._coords + "/origin/y"],
                        self._node["coordsets/" + self._coords + "/origin/z"])
                    spacing = (
                        self._node["coordsets/" +
                                   self._coords + "/spacing/dx"],
                        self._node["coordsets/" +
                                   self._coords + "/spacing/dy"],
                        self._node["coordsets/" +
                                   self._coords + "/spacing/dz"])
                    oi.Set(vtkDataObject.ORIGIN(), origin, 3)
                    oi.Set(vtkDataObject.SPACING(), spacing, 3)
                    (self._whole_extent, self._extent) = extents_uniform(
                        self._node, self._comm, self._mpi_size, self._mpi_rank,
                        self._topology)
                elif (self._outputType == vtkConstants.VTK_RECTILINEAR_GRID):
                    (self._whole_extent, self._extent) = extents_rectilinear(
                        self._node, self._comm, self._mpi_size, self._mpi_rank,
                        self._topology)
                elif (self._outputType == vtkConstants.VTK_STRUCTURED_GRID):
                    dims = (self._node["topologies/" + self._topology +
                                       "/elements/dims/i"] + 1,
                            self._node["topologies/" + self._topology +
                                       "/elements/dims/j"] + 1,
                            self._node["topologies/" + self._topology +
                                       "/elements/dims/k"] + 1)
                    self._extent = (
                        0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
                    self._whole_extent = self._extent
                oi.Set(vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(),
                       self._whole_extent, 6)
            oi.Set(vtkAlgorithm.CAN_HANDLE_PIECE_REQUEST(), 1)
        return 1

    def FillOutputPortInformation(self, port, info):
        self._outputType = None
        self._topology = self._topologies[port]
        self._coords = self._node["topologies/" + self._topology +
                                  "/coordset"]
        topologyType = self._node["topologies/" + self._topology + "/type"]
        if topologyType == "uniform":
            self._outputType = vtkConstants.VTK_IMAGE_DATA
            self._extension = "vti"
            outputTypeName = "vtkImageData"
        elif topologyType == "rectilinear":
            self._outputType = vtkConstants.VTK_RECTILINEAR_GRID
            self._extension = "vtr"
            outputTypeName = "vtkRectilinearGrid"
        elif topologyType == "structured":
            self._outputType = vtkConstants.VTK_STRUCTURED_GRID
            self._extension = "vts"
            outputTypeName = "vtkStructuredGrid"
        elif topologyType == "unstructured":
            self._outputType = vtkConstants.VTK_UNSTRUCTURED_GRID
            self._extension = "vtu"
            outputTypeName = "vtkUnstructuredGrid"
        else:
            print("Error: invalid topology type {}".format(
                topologyType))
            return 0
        info.Set(vtkDataObject.DATA_TYPE_NAME(), outputTypeName)
        return 1

    def GetOutputType(self):
        ''' Returns the output vtkDataObject type.
        '''
        return self._outputType

    @smproperty.xml("""
        <StringVectorProperty
            name="OutputFileExtension"
            number_of_elements="1"
            information_only="1"
            command="GetOutputFileExtension">
        </StringVectorProperty>""")
    def GetOutputFileExtension(self):
        return self._extension
