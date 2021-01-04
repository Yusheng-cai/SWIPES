import swipes.isosurface as prod_iso 
from isosurface import *
import numpy as np
import time
import multiprocessing as mp
from analysis_code.Liquid_crystal.Liquid_crystal import *
import os

def find_pos(LC_obj,time,constraints):
          xmin,xmax,ymin,ymax,zmin,zmax = constraints
          pos = LC_obj.COM(time,'whole')
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
          LC_obj = LC(os.getcwd()+"/SWIPES5CB_test",10,5,bulk=False)
          constraints = np.array([110,210,0,70,32,113.5])

          pos = find_pos(LC_obj,0,constraints)
          box = np.array([100,70,80])
          ngrids = np.array([50,50,50])
          iso = prod_iso(box,ngrids,sigma=6,n=3,kdTree=False)
          iso_test = isosurface(box,ngrids,sigma=6,n=3,kdTree=False) 

          s = time.perf_counter()
          iso.field_density_cube(pos,keep_d=np.array([0,1,1]))
          e = time.perf_counter()
          print("Time it takes for JIT compiled CG fcn is {}".format(e-s)) 

          s = time.perf_counter()
          iso_test.field_density_cube(pos,keep_d=np.array([0,1,1]))
          e = time.perf_counter()
          print("Time it takes for non JIT compiled CG fcn is {}".format(e-s)) 
 

          