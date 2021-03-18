import sys
sys.path.insert(0,"..")
from isosurface import isosurface
import numpy as np
import pymp
import MDAnalysis as mda
import time

def run(t,u,iso):
    ix = 0
    u.trajectory[t]
    OW = u.select_atoms("type OW")
    pos = OW.atoms.positions

    pos = pos[pos[:,0]>=xmin]
    pos = pos[pos[:,0]<=xmax]

    pos = pos[pos[:,1]>=ymin]
    pos = pos[pos[:,1]<=ymax]

    pos = pos[pos[:,2]>=zmin]
    pos = pos[pos[:,2]<=zmax]

    pos[:,0] -= pos[:,0].min()
    pos[:,1] -= pos[:,1].min()
    pos[:,2] -= pos[:,2].min()

    f = iso.field_density_cube(pos,d=np.array([0,1,1]))
    print("{} is done".format(t))
    return f

if __name__ == '__main__':
    start = 0
    end = 100
    N = 100


    xmin = 70.0
    xmax = 150.0
    Lx = xmax - xmin

    ymin = 0.0
    ymax = 70.0
    Ly = ymax - ymin

    zmin = 35.0
    zmax = 60.0
    Lz = zmax - zmin

    mesh_x = 50
    mesh_y = 50
    mesh_z = 50
    mesh = np.array([mesh_x,mesh_y,mesh_z])
    box = np.array([Lx,Ly,Lz])
    
    field = pymp.shared.array((mesh_x,mesh_y,mesh_z))
    s = time.time()
    with pymp.Parallel(4) as p:
        iso = isosurface(box,mesh,n=2.5,kdTree=False)
        u = mda.Universe("../test/SWIPES_2900/SWIPES_2900.tpr","../test/SWIPES_2900/SWIPES_2900_pbc.xtc")
        for t in p.range(start,end):
            field += run(t,u,iso)/N

    np.save("mp_result.npy",field)

    e = time.time()

    print(e-s)
