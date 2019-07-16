###############################################################################
#
# file: ascent_tutorial_demo_5_rotation.py
#
###############################################################################
import os
import numpy as np
from mpi4py import MPI
from conduit import Node
import ascent.mpi

class myAscent():
    ''' Class Structure for ascent '''
    def __init__(self,
                 resolution=100,
                 rho=20,
                 rootdir="./output",
                 useMPI=True,
                 ):
        self.comm      = MPI.Comm.f2py(ascent_mpi_comm_id())
        self.mesh_data = ascent_data().child(0)

        if useMPI:
            self._ascent = ascent.mpi.Ascent()
        else:
            self._ascent = ascent.Ascent()

        self.cycle      = self.mesh_data["state/cycle"]
        #self.timestep   = self.mesh_data["state/time"]
        if "/" not in rootdir[-1]:
            rootdir += "/"
        self.rootdir = rootdir
        if rootdir[-1] == "/":
            self.rootdir = rootdir
        else:
            self.rootdir = rootdir + "/"
        self.resolution = resolution
        self.rho        = rho
        self.theta      = np.linspace(0,2*np.pi,self.resolution)
        self.scenes     = Node()
        self.extracts   = Node()

    def __del__(self):
        ''' destructor to make sure ascent closes nicely '''
        if self._ascent:
            self._ascent.close()

    def center(self):
        '''
        We want to set the origin of our cube to be the exact center. This
        will allow us to cleanly rotate around the object
        '''
        # Make origin at the center of the cube
        x = self.mesh_data["coordsets/coords/values/x"]
        y = self.mesh_data["coordsets/coords/values/y"]
        z = self.mesh_data["coordsets/coords/values/z"]
        
        # Conduit data must be numpy arrays
        self.mesh_data["coordsets/coords/values/x"] = np.asarray([xx - max(x)/2 for xx in x], dtype=np.float64)
        self.mesh_data["coordsets/coords/values/y"] = np.asarray([yy - max(y)/2 for yy in y], dtype=np.float64)
        self.mesh_data["coordsets/coords/values/z"] = np.asarray([zz - max(z)/2 for zz in z], dtype=np.float64)
    
    def create_views(self):
        '''
        We'll use cylindrical coordinates for an easy rotation about the y axis
        '''
        x     = [self.rho*np.cos(t) for t in self.theta]
        y     = [self.rho*np.sin(t) for t in self.theta]
        views = np.asarray([[xx,0,yy] for xx,yy in zip(x,y)], dtype = np.float64)
        return views

    def make_renders(self):
        renders = Node()
        views = self.create_views()
        for i,view in enumerate(views):
            r = "r" + str(i)
            name = "theta_" + str(self.theta[i])[:4] + "_"
            renders[r+"/annotations"] = "false"
            pathname = self.rootdir + str(self.cycle)
            if not os.path.exists(pathname):
                os.system('mkdir -p ' + pathname)
            renders[r+"/image_name"] = pathname + "/" + name
            renders[r+"/camera/position"] = view
        self.scenes["s1/renders"] = renders

    def make_scenes(self):
        ''' Function to hold our scenes '''
        self.scenes["s1/plots/p1/type"] = "pseudocolor"
        self.scenes["s1/plots/p1/field"] = "energy"
        self.make_renders()

    def make_actions(self):
        ''' Function for our actions '''
        actions = Node()
        add_act = actions.append()
        add_act["action"] = "add_scenes"
        add_act["scenes"] = self.scenes
        add_act = actions.append()
        # New actions for the extracts
        add_act["action"] = "add_extracts"
        add_act["extracts"] = self.extracts
        actions.append()["action"] = "execute"
        self._ascent.execute(actions)

    def make_hdf5(self):
        ''' Saves data to hdf5 '''
        self.extracts["e2/type"] = "relay"
        pathname = str(self.rootdir) + str(self.cycle) + "/data"
        if not os.path.exists(pathname):
            os.system('mkdir -p ' + str(pathname))
        assert(os.path.exists(pathname))
        self.extracts["e2/params/path"] = pathname + "/rotation_data"
        self.extracts["e2/params/protocol"] = "blueprint/mesh/hdf5"

    def run(self):
        ''' Run everything together '''
        ascent_opts = Node()
        ascent_opts["actions_file"] = ""
        ascent_opts["mpi_comm"].set(self.comm.py2f())
        self._ascent.open(ascent_opts)
        self._ascent.publish(self.mesh_data)
        self.make_hdf5()
        self.make_scenes()
        self.make_actions()

if __name__ == '__main__':
    a = myAscent(resolution=10,
            rootdir="myPath",
            )
    a.run()
