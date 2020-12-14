import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

class isosurface:
    def __init__(self,box,ngrids,sigma=2.4):
        """
        pos: position of the atoms/virtual atoms(COM) in the desired probe volume (N,3),
             these are already normalized where pox[x,y,z] all lie respectively in [0,Lx),[0,Ly),[0,Lz)
        box: a tuple of (Lx,Ly,Lz)
        spacings: a tuple of (dx,dy,dz)
        sigma: the sigma used for coarse graining of density field
        """
        self.box = box
        self.Lx,self.Ly,self.Lz = box
        self.nx,self.ny,self.nz = ngrids
        self.sigma = sigma
        self.field = None
        self.initialize()

    def initialize(self):
        X = np.linspace(0,self.Lx,num=self.nx,endpoint=False)
        Y = np.linspace(0,self.Ly,num=self.ny,endpoint=False)
        Z = np.linspace(0,self.Lz,num=self.nz,endpoint=False)
        
        xx,yy,zz = np.meshgrid(X,Y,Z) # each of xx,yy,zz are of shape (Ni,Ni,Ni)
        xx = np.moveaxis(xx,0,-1)
        yy = np.moveaxis(yy,1,0)
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

    def field_density(self,pos,n=2.5,ignore_d=None):
        """
        This is not a exact way to find the density field, but cut off the density gaussian at 
        n*sigma 

        n: the n in the cutoff n*sigma that we want to approximate the density field by
        ignore_d: the dimension to be ignored in pbc calculation, a numpy array with shape (3,)

        returns:
            the field density 
        """
        sigma = self.sigma
        tree = self.tree
        box = self.box
        grids = self.grids
        ignore_d_flag = False
        if isinstance(ignore_d,np.ndarray):
            d = ignore_d
            ignore_d_flag = True

        self.idx = tree.query_ball_point(pos,sigma*n)  
        self.field = np.zeros((self.nx*self.ny*self.nz,))

        ix = 0
        for index in self.idx:
            dr = abs(pos[ix] - grids[index])
            # check pbc
            cond = 1*(dr > box/2)
            if ignore_d_flag:
                cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)
            self.field[index] += self.coarse_grain(dr,sigma)
            ix += 1

        return self.field

    def marching_cubes(self,c=0.016,gradient_direction='descent'):
        """
        Output triangles needed for graphing isosurface 
        c: the contour line value for the isosurface
        gradient_direction: 'descent' if the values exterior of the object are smaller,
                            'ascent' if the values exterior of the object are bigger
        
        output: 
                the indices for all triangles (N,3,3) where N=number of triangles
        """
        if self.field is None:
            raise RuntimeError("Please run iso.field_density first!")

        Nx,Ny,Nz = self.nx,self.ny,self.nz
        Lx,Ly,Lz = self.Lx,self.Ly,self.Lz
        dx,dy,dz = Lx/Nx,Ly/Ny,Lz/Nz

        field = self.field
        data = field.reshape(Nx,Ny,Nz)
        verts,faces,_,_ = measure.marching_cubes_lewiner(data,c,spacing=(dx,dy,dz))

        return verts[faces]
