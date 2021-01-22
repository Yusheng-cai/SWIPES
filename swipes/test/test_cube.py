import sys
sys.path.insert(0,"..")
from isosurface import isosurface
import numpy as np
import MDAnalysis as mda

def find_pos(u,t,constraints):
    xmin,xmax,ymin,ymax,zmin,zmax = constraints
    u.trajectory[t]
    waters = u.select_atoms("type OW")
    pos = waters.atoms.positions

    pos = pos[pos[:,0]>=xmin]
    pos = pos[pos[:,0]<=xmax]

    pos = pos[pos[:,1]>=ymin]
    pos = pos[pos[:,1]<=ymax]

    pos = pos[pos[:,2]>=zmin]
    pos = pos[pos[:,2]<=zmax]

    pos[:,0] -= pos[:,0].min()
    pos[:,1] -= pos[:,1].min()
    pos[:,2] -= pos[:,2].min()

    return pos

def test_cube():
    t = 5000

    u = mda.Universe("SWIPES_2900/SWIPES_2900.tpr","SWIPES_2900/SWIPES_2900_pbc.xtc")
    constraints = np.array([70,155,0,70,35,60])

    pos = find_pos(u,t,constraints)
    box = np.array([85,70,25])
    ngrids = np.array([50,50,50])

    iso = isosurface(box,ngrids,kdTree=False)
    field = iso.field_density_cube(pos,d=np.array([0,1,1]))
    correct_field = np.load("SWIPES_2900/SWIPES_2900_cube.npy")
    assert np.linalg.norm((field-correct_field).flatten(),2) < 1e-10
