import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree

class isosurface:
    def __init__(self,pos,box,ngrids,sigma=2.4):
        """
        pos: position of the atoms/virtual atoms(COM) in the desired probe volume (N,3),
             these are already normalized where pox[x,y,z] all lie respectively in [0,Lx),[0,Ly),[0,Lz)
        box: a tuple of (Lx,Ly,Lz)
        spacings: a tuple of (dx,dy,dz)
        sigma: the sigma used for coarse graining of density field
        """
        self.pos = pos
        self.box = box
        self.Lx,self.Ly,self.Lz = box
        self.nx,self.ny,self.nz = ngrids
        self.sigma = sigma
        self.initialize()

    def initialize(self):
        X = np.linspace(0,self.Lx,num=self.nx,endpoint=False)
        Y = np.linspace(0,self.Ly,num=self.ny,endpoint=False)
        Z = np.linspace(0,self.Lz,num=self.nz,endpoint=False)
        
        xx,yy,zz = np.meshgrid(X,Y,Z) # each of xx,yy,zz are of shape (Ni,Ni,Ni)
        self.grids = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T
        self.tree = cKDTree(self.grids,boxsize=self.box)
    
    def coarse_grain(self,dr,sigma):
        """
        coarse graining function for the density of a field
        dr: the vector distance (could be float, 1d np.ndarray vector or 2d np.ndarray matrix)
        sigma: the "standard deviation" of the gaussian field applied on each of the molecules
        
        returns:
            the coarse grained density (float, 1d np.ndarray or 2d np.ndarray that matches the input) 
        """
        if isinstance(dr,np.ndarray):
            if dr.ndim >= 2:
                d = dr.shape[-1]
                sum_ = (dr**2).sum(axis=-1)
            if dr.ndim == 1:
                d = dr.shape[0]
                sum_ = (dr**2).sum()

        if isinstance(dr,float) or isinstance(dr,int):
            d = 1
            sum_ = dr**2

        return (2*np.pi*sigma**2)**(-d/2)*np.exp(-sum_/(2*sigma**2))

    def field_density(self,n=2.5):
        """
        This is not a exact way to find the density field, but cut off the density gaussian at 
        n*sigma 

        n: the n in the cutoff n*sigma that we want to approximate the density field by

        returns:
            the field density 
        """
        pos = self.pos
        sigma = self.sigma
        tree = self.tree
        box = self.box
        grids = self.grids

        self.idx = tree.query_ball_point(pos,sigma*n)  
        self.field = np.zeros((self.nx*self.ny*self.nz,))

        ix = 0
        for index in self.idx:
            dr = abs(pos[ix] - grids[index])
            # check pbc
            cond = dr > box/2

            # correct pbc
            dr = abs(cond*box - dr)
            self.field[index] += self.coarse_grain(dr,sigma)
            ix += 1

        return self.field
