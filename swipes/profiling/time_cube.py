import time
import multiprocessing as mp
import timeit
import os

def find_pos(u,time,constraints):
      u.trajectory[time] 
      nCB = u.select_atoms("resname 5CB")
      pos = nCB.atoms.positions

      xmin,xmax,ymin,ymax,zmin,zmax = constraints
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

if __name__ == '__main__':
    SETUP_CODE = '''
import sys
sys.path.insert(0,'/Users/caiyusheng/Desktop/Research/Research_code/SWIPES/swipes')
from isosurface import isosurface
from analysis_code.Liquid_crystal.Liquid_crystal import LC
import numpy as np
import os 
from __main__ import find_pos
LC_obj = LC("../test/SWIPES5CB_16400",10,5,bulk=False)
constraints = np.array([130,210,0,70,32,113.5])

u = LC_obj["universe"]
pos = find_pos(u,0,constraints)
print(pos.shape)
box = np.array([100,70,80])
ngrids = np.array([50,50,50])

iso = isosurface(box,ngrids,sigma=3.4,n=3,kdTree=False)
'''

    RUN_CODE = '''
iso.field_density_cube(pos,keep_d=np.array([0,1,1]))
'''

    print(timeit.timeit(setup=SETUP_CODE,stmt=RUN_CODE,number=10)/10)

