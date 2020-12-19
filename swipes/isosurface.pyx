import MDAnalysis as mda
import numpy as np
import time
from scipy.spatial import cKDTree
from skimage import measure

class isosurface:
    def __init__(self,box,ngrids,sigma=2.4,kdTree=True,field=None):
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

        # User can pass in a field or else it will be None
        if field is not None:
            print("You have passed in a density field!")
        self.field = field
        self.dict = {}

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
            print("KDtree built, now isosurface.field_density_kdtree can be used.")
        else:
            self.tree = None
    
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
        if self.field is not None:
            print("The field that was passed in will now be overwritten")
        sigma = self.sigma

        if self.tree is None:
            raise RuntimeError("Please set kdTree=True in initialization, tree has not been built")
        else:
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

    def field_density_cube(self,pos,n=2.5,keep_d=None,verbose=False):
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
        if self.field is not None:
            if verbose:
                print("The field that was passed or was just calculated in will now be overwritten")

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
            # adding dictionary actually decreases performance because of memory usage
            # num = indices[-1]*Nx*Ny+indices[1]*Nx+indices[0]
            #if num in self.dict:
            #    idx = self.dict[num]
            #    if verbose:
            #        print("dict used")
            #else:
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
            #self.dict[num] = idx
            dr = abs(p - grids[idx[:,0],idx[:,1],idx[:,2]])

            # check pbc
            cond = 1*(dr > box/2)
            if keep_d_flag:
                cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)

            self.field[idx[:,0],idx[:,1],idx[:,2]] += self.coarse_grain(dr,sigma)

        return self.field

    def surface1d(self,field1d=None,grids1d=None,c=0.016):
        """
        Function that calculates the points to plot a 1d surface

        field1d: the x field usually (Nx,), if None, then will be calculated from self.field by integrating
        out y and z dependence
        grids1d: the x grids usually (Nx,), if None, then will generate grids based on self.Nx and self.Lx
        c: where the isosurface lie

        returns:
           The point where the surface cross 0.016 
        """
        if field1d is None and self.field is None:
            raise RuntimeError("Please provide a field or run self.field_density_cube or self.density_kdtree!")

        if field1d is not None:
            field = field1d
        else:
            field = (self.field.sum(axis=-1)/self.nz).sum(axis=-1)/self.ny # (Nx,)

        if grids1d is None:
            grids = np.linspace(0,self.Lx,self.nx)
        else:
            grids = grids1d

        if len(field.shape) != 1:
            raise RuntimeError("Please provide a 1d field in the shape of (Nx,)!")
      
        pfield = field[0]
        pgrid = grids[0]
        for i in range(1,field.shape[0]):
            cfield = field[i]
            cgrid = grids[i]

            cond1 = pfield >= c
            cond2 = cfield <= c
            cond = cond1*cond2

            if cond != 0:
                ratio = (c - pfield)/(cfield-pfield) 
                x = pgrid + ratio*(cgrid-pgrid)

            pfield = cfield
            pgrid = cgrid
        return x

    def surface2d(self,field2d=None,grids2d=None,c=0.016):
        """
        Function that calculates the points to plot a 2d surface
        fields2d: the x & z field usually (Nx,Nz), if None, then will use self.field
        grids2d: the x & z grids usually, if None, then will generate grids
        c: where the isosurface lie

        returns:    
            a list of points that lies on the 2d isosurface
        """
        if field2d is None and self.field is None:
            raise RuntimeError("Please provide a field or run self.field_density_cube or self.density_kdtree!")

        if field2d is not None:
            field = field2d
        else:
            field = self.field.sum(axis=1)/self.ny # (Nx,Ny)

        if grids2d is None:
            x = np.linspace(0,self.Lx,self.nx)
            z = np.linspace(0,self.Lz,self.nz)

            xx,zz = np.meshgrid(x,z)
            xx = np.moveaxis(xx,0,-1)
            zz = np.moveaxis(zz,0,-1)
            grids = np.concatenate((xx[:,:,np.newaxis],zz[:,:,np.newaxis]),axis=-1) # (Nx,Nz,2)
        else:
            grids = grids2d

        if len(field.shape) != 2:
            raise RuntimeError("Please provide a 2d field in the shape of (Nx,Nz)!")

        Nx,Nz = field.shape
        # define "previous field" as in all the field that has the same x coordinate (in this case x=0)
        pfield = field[0]
        # define "previous grids" as in all the grids that has the same x coordinate (in this case x=0)
        pgrids = grids[0]
        points_all = []

        for i in range(1,Nx):
            # define "current field/grids" as in the current grid/field we are looking at that share the same x coordinate
            cfield = field[i] # (nz, )
            cgrids = grids[i] # (nz,2)
            
            # first condition is the previous x (field) needs to be larger than c
            cond1 = pfield >= c
            # second condition is the current x (field) needs to be smaller than c
            cond2 = cfield <= c
            # both have to be true in order to be identified to be at the boundary
            cond = cond1*cond2
            # find the indices of the points that satisfy both conditions
            idx = np.argwhere(cond == 1)
            
            # See if idx is empty (only perform when it is not)
            if idx.size:
                # find where c lies between the current field and previous field
                ratio = (c-pfield[idx])/(cfield[idx]-pfield[idx])

                # find current x and previous x
                x_prev = pgrids[idx,0]
                x_curr = cgrids[idx,0]

                z_points = grids[i,idx,1]
                # interpolate the x values on the boundary by the field values
                x_points = x_prev + (x_curr - x_prev)*ratio

                # stack the x and z values to form the 2d isosurface
                points = np.vstack((x_points.flatten(),z_points.flatten())).T
                points_all.append(points)
            
            # update "previous" field/grids
            pfield = cfield
            pgrids = cgrids

        return np.concatenate(points_all)

 
    def surface3d(self,c=0.016,gradient_direction='descent',field=None):
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
