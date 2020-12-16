import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

class isosurface:
    def __init__(self,box,ngrids,sigma=2.4,kdTree=True):
        """
        pos: position of the atoms/virtual atoms(COM) in the desired probe volume (N,3),
             these are already normalized where pox[x,y,z] all lie respectively in [0,Lx),[0,Ly),[0,Lz)
        box: a np.ndarray of [Lx,Ly,Lz]
        ngrids: a np.ndarray of [Nx,Ny,Nz]
        sigma: the sigma used for coarse graining of density field
        """
        self.box = box
        self.Lx,self.Ly,self.Lz = box

        self.ngrids = ngrids
        self.nx,self.ny,self.nz = ngrids

        self.dbox = box/ngrids
        self.dx,self.dy,self.dz = self.dbox

        self.sigma = sigma

        # set the initial field to None
        self.field = None
        self.initialize(kdTree)

    def initialize(self,kdTree=True):
        X = np.linspace(0,self.Lx,num=self.nx,endpoint=False)
        Y = np.linspace(0,self.Ly,num=self.ny,endpoint=False)
        Z = np.linspace(0,self.Lz,num=self.nz,endpoint=False)
        
        xx,yy,zz = np.meshgrid(X,Y,Z) # each of xx,yy,zz are of shape (Ni,Ni,Ni)
        xx = np.moveaxis(xx,0,-1)
        yy = np.moveaxis(yy,1,0)
        self.grids = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T
        if kdTree:
            self.tree = cKDTree(self.grids,boxsize=self.box)
            print("Using KDTree algorithm")
    
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

    def field_density_kdtree(self,pos,n=2.5,keep_d=None):
        """
        This is not a exact way to find the density field, but cut off the density gaussian at 
        n*sigma. The meshgrid points within the radius are found using kdtree, building of the 
        tree is M log(M) and searching takes log(M). 

        n: the n in the cutoff n*sigma that we want to approximate the density field by
        keep_d: the dimension to be ignored in pbc calculation, a numpy array with shape (3,)

        returns:
            the field density 
        """
        sigma = self.sigma
        tree = self.tree
        box = self.box
        grids = self.grids
        keep_d_flag = False
        Nx,Ny,Nz = self.nx,self.ny,self.nz

        if isinstance(keep_d,np.ndarray):
            d = keep_d
            keep_d_flag = True

        self.idx = tree.query_ball_point(pos,sigma*n)  
        self.field = np.zeros((self.nx*self.ny*self.nz,))

        ix = 0
        for index in self.idx:
            dr = abs(pos[ix] - grids[index])
            # check pbc
            cond = 1*(dr > box/2)
            if keep_d_flag:
                cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)
            self.field[index] += self.coarse_grain(dr,sigma)
            ix += 1

        self.field = self.field.reshape(Nx,Ny,Nz)

        return self.field

    def field_density_cube(self,pos,n=2.5,keep_d=None):
        """
        Find all the distances in a cube, this method doesn't use any search method but rather indexing into self.grids array
        For every atom, it first finds the nearest index to the atom by simply perform floor(x/dx,y/dy,z/dz). Once the nearest
        index to the atom is found which we will call (ix,iy,iz), it then tries to find all the points in a cube with length 2L 
        centered around the index. L is usually determined by n*sigma. 
        The number of indices that one needs to search in every direction is determined by ceil(L/dx,L/dy,L/dz) which we call (nx,ny,nz). 
        So xrange=(ix-nx,ix+nx), yrange=(iy-ny,iy+ny),zrange=(iz-nz,iz+nz). All the indices can then be found by 
        meshgrid(xrange,yrange,zrange). Then PBC can be easily taken care of by subtracting all indices in the meshgrid that are larger than (nx,ny,nz)
        by (nx,ny,nz) and add (nx,ny,nz) to all the ones that are smaller than 0.

        pos: the positions of the atoms (Ntot,3)
        n: the "radius" of a cube that the code will search for 
        keep_d: which dimension will not be ignored (numpy array (3,))

        returns: 
                a field of shape (Nx,Ny,Nz) from ngrids
        """
        dbox = self.dbox
        ngrids = self.ngrids
        box = self.box
        sigma = self.sigma
        keep_d_flag = False

        Nx,Ny,Nz = self.nx,self.ny,self.nz
        dx,dy,dz = self.dx,self.dy,self.dz
        # create grids and empty field
        grids = self.grids.reshape((Nx,Ny,Nz,3))
        self.field = np.zeros((Nx,Ny,Nz))

        # the length of the cubic box to search around an atom
        L = n*sigma

        # the number of index to search in [x,y,z]
        nidx_search = np.ceil(L/dbox) 

        if isinstance(keep_d,np.ndarray):
            d = keep_d
            keep_d_flag = True

        for p in pos: 
            indices = np.ceil(p/dbox)

            back = indices - nidx_search
            forward = indices + nidx_search

            x = np.r_[int(back[0]):int(forward[0])+1]
            y = np.r_[int(back[1]):int(forward[1])+1]
            z = np.r_[int(back[2]):int(forward[2])+1]

            xx,yy,zz = np.meshgrid(x,y,z)
            idx = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T
            left_PBC_cond = 1*(idx < 0)
            right_PBC_cond = 1*(idx > ngrids-1)

            idx += left_PBC_cond*ngrids
            idx -= right_PBC_cond*ngrids

            dr = abs(p - grids[idx[:,0],idx[:,1],idx[:,2]])

            # check pbc
            cond = 1*(dr > box/2)
            if keep_d_flag:
                cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)

            self.field[idx[:,0],idx[:,1],idx[:,2]] += self.coarse_grain(dr,sigma)

        return self.field
     
    def marching_cubes(self,c=0.016,gradient_direction='descent',field=None):
        """
        Output triangles needed for graphing isosurface 
        c: the contour line value for the isosurface
        gradient_direction: 'descent' if the values exterior of the object are smaller,
                            'ascent' if the values exterior of the object are bigger
        
        output: 
                the indices for all triangles (N,3,3) where N=number of triangles
        """
        if self.field is None and field is None:
            raise RuntimeError("Please run iso.field_density first or pass in a field!")

        if field is not None:
            field=field
        else:
            field =self.field
        

        Nx,Ny,Nz = self.nx,self.ny,self.nz
        Lx,Ly,Lz = self.Lx,self.Ly,self.Lz
        dx,dy,dz = self.dx,self.dy,self.dz

        verts,faces,_,_ = measure.marching_cubes_lewiner(field,c,spacing=(dx,dy,dz))

        return verts[faces] 
