import sys
sys.path.insert(0,"..")
from isosurface import isosurface
import numpy as np
import MDAnalysis as mda
import time

def run(u,iso,field,timestep):
    ix = 0
    for t in timestep:
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

        field += iso.field_density_cube(pos,d=np.array([0,1,1]))
        print("{} is done".format(t))
        ix += 1
    return field

if __name__ == '__main__':
    start = 0
    end = 100
    N = 100

    times = np.arange(start,end)

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

    iso = isosurface(box,mesh,n=2.5,kdTree=False)
    print("isosurface object made")


    field = np.zeros((mesh_x,mesh_y,mesh_z)) 
    s = time.time()

    u = mda.Universe("../test/SWIPES_2900/SWIPES_2900.tpr","../test/SWIPES_2900/SWIPES_2900_pbc.xtc")
    f = run(u,iso,field,times)/N

    e = time.time()
    np.save("nomp_result.npy",f)
    print(e-s)
