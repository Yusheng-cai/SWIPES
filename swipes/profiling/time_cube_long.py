from swipes.isosurface import *
from analysis_code.Liquid_crystal.Liquid_crystal import *
from wham.lib.utils import *
import numpy as np
import multiprocessing as mp
import os
import time

def run(u,iso,field,timestep,output):
    ix = 0
    N = len(timestep)
    for t in timestep:
        u.trajectory[t]
        nCB = u.select_atoms("resname 5CB")
        pos = nCB.atoms.positions
        pos = pos[pos[:,0]>=xmin]
        pos = pos[pos[:,0]<=xmax]

        pos = pos[pos[:,1]>=ymin]
        pos = pos[pos[:,1]<=ymax]

        pos = pos[pos[:,2]>=zmin]
        pos = pos[pos[:,2]<=zmax]

        pos[:,0] -= pos[:,0].min()
        pos[:,1] -= pos[:,1].min()
        pos[:,2] -= pos[:,2].min()

        field += iso.field_density_cube(pos,keep_d=np.array([0,1,1]))/N
        print("{} is done".format(t))
        ix += 1
    output.put(field)

if __name__ == '__main__':
    path = '../test/SWIPES5CB_16400'
    N = "test"
    c = 0.00125
    start = 0
    end = 40
    skip = 1
    num_process = 8

    times = np.arange(start,end,step=skip)
    times_list = np.array_split(times,num_process)

    xmin = 130.0
    xmax = 210.0
    Lx = xmax - xmin

    ymin = 0.0
    ymax = 70.0
    Ly = ymax - ymin

    zmin = 32.0
    zmax = 113.5
    Lz = zmax - zmin

    mesh_x = 50
    mesh_y = 50
    mesh_z = 50
    mesh = np.array([mesh_x,mesh_y,mesh_z])
    box = np.array([Lx,Ly,Lz])

    iso = isosurface(box,mesh,sigma=3.4,n=3,kdTree=False)
    print("isosurface object made with kdtree")


    field = np.zeros((mesh_x,mesh_y,mesh_z))
    output = mp.Queue()

    process = []
    

    for i in range(num_process):
        u = LC(path,10,5,bulk=False)['universe']
        p = mp.Process(target=run,args=(u,iso,field,times_list[i],output))
        process.append(p)

    s = time.time()
    for p in process:
        p.start()
    
    
    for p in process:
        field += output.get()/num_process

    for p in process:
        p.join()
    e = time.time()
    print(e-s)
