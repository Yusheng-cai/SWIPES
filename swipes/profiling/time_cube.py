import time
import multiprocessing as mp
import timeit
import os

def find_pos(u,time,constraints):
      u.trajectory[time] 
      OH = u.select_atoms("type OW")
      pos = OH.atoms.positions

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
sys.path.insert(0,"..")
from isosurface import isosurface
import MDAnalysis as mda
import numpy as np
import os 
from __main__ import find_pos

u = mda.Universe("../test/SWIPES_2900/SWIPES_2900.tpr","../test/SWIPES_2900/SWIPES_2900_pbc.xtc")
constraints = np.array([70,155,0,70,35,60])
pos = find_pos(u,0,constraints)

box = np.array([155-70,70,25])
ngrids = np.array([50,50,50])

iso = isosurface(box,ngrids,n=2.5,kdTree=False)
'''

    RUN_CODE = '''
iso.field_density_cube(pos,d=np.array([0,1,1]))
'''

    print(timeit.timeit(setup=SETUP_CODE,stmt=RUN_CODE,number=10)/10)

