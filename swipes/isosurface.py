import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from numba import jit,njit
       
class isosurface:
    """
    Args:
        box(np.ndarray): An array of [Lx,Ly,Lz]
        ngrids(np.ndarray): Number of grid points in each direction passed in as [Nx,Ny,Nz]
        sigma(np.ndarray): the sigma used for coarse graining function of density field by Willard and Chandler (default 2.4 for water)
        n(float): The cutoff radius for n*sigma (default 2.5 for water)
        kdtree(bool): whether or not to build kdtree (default True)
        verbose(bool): whether or not to print stuff (default False)
    """
    def __init__(self,box,ngrids,sigma=2.4,n=2.5,kdTree=True,verbose=False):
        self.verbose = verbose
        # Length of the box
        self.box = box
        self.Lx,self.Ly,self.Lz = box
     
        # number of grids each way
        self.ngrids = ngrids
        self.nx,self.ny,self.nz = ngrids
        
        # spacing between grids  
        self.dbox = box/ngrids
        self.dx,self.dy,self.dz = self.dbox
        
        # other parameters
        self.sigma = sigma
        self.n = n
        self.L = n*sigma
        self.nidx_search = np.ceil(self.L/self.dbox) 
        self.initialize(kdTree)

    def initialize(self,kdTree=True):
        """
        Function that initializes the isosurface class

        Args:
            kdtree(bool): A boolean that defines whether or not the method uses kdtree algorithm
        Return:
            self.ref_idx(np.ndarray): Imagine a point is surrounded by a cube volume of grid points, ref_idx is the grid points sorrounding point (0,0,0) where then the latter ones can just simply add (x,y,z) to this to obtain the points in their respective cube 
            self.grids(np.ndarray): the (x,y,z) grids shaped in (Nx,Ny,Nz) 
            
        """
        X = np.linspace(0,self.Lx,num=self.nx,endpoint=False)
        Y = np.linspace(0,self.Ly,num=self.ny,endpoint=False)
        Z = np.linspace(0,self.Lz,num=self.nz,endpoint=False)
        
        xx,yy,zz = np.meshgrid(X,Y,Z) # each of xx,yy,zz are of shape (Ni,Ni,Ni)

        # Doing this to ensure that when reshape back, self.grids will be at the correct order in (X,Y,Z)
        xx = np.moveaxis(xx,0,-1)
        yy = np.moveaxis(yy,1,0)
        self.grids = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

        if kdTree:
            self.tree = cKDTree(self.grids,boxsize=self.box)
            if self.verbose:
                print("KDtree built, now isosurface.field_density_kdtree can be used.")
        else:
            self.tree = None
        
        # find reference "cube" for field_density_cube
        ref_indices = np.array([0,0,0])
        back = ref_indices - self.nidx_search
        forward = ref_indices + self.nidx_search

        x = np.r_[int(back[0]):int(forward[0])+1]
        y = np.r_[int(back[1]):int(forward[1])+1]
        z = np.r_[int(back[2]):int(forward[2])+1]
        xx,yy,zz = np.meshgrid(x,y,z)

        # index surrounding the point of interest
        self.ref_idx = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

    
    def field_density_kdtree(self,pos,d=np.array([1,1,1])):
        """
        This is not a exact way to find the density field, but cut off the density gaussian at 
        n*sigma. The meshgrid points within the radius are found using kdtree, building of the 
        tree is M log(M) and searching takes log(M). 

        Args:
            pos(np.ndarray): position of the atoms/virtual atoms(COM) in the desired probe volume of shape (N,3),these should be already normalized where pos[0,:], pos[1,:],pos[2:,] all lie respectively in [0,Lx),[0,Ly),[0,Lz)
            d(np.ndarray): the dimension to be kept in pbc calculation. (3,) (default [1,1,1])

            returns:
                the field density (Nx,Ny,Nz)
        """
        sigma = self.sigma

        if self.tree is None:
            raise RuntimeError("Please set kdTree=True in initialization, tree has not been built")
        else:
            tree = self.tree

        box = self.box
        grids = self.grids
        Nx,Ny,Nz = self.nx,self.ny,self.nz
        n = self.n

        self.idx = tree.query_ball_point(pos,sigma*n)  
        self.field = np.zeros((self.nx*self.ny*self.nz,))

        ix = 0
        for index in self.idx:
            dr = abs(pos[ix] - grids[index])

            # check pbc
            cond = 1*(dr > box/2)
            cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)
            self.field[index] += coarse_grain(dr, sigma)
            ix += 1                 

        self.field = self.field.reshape(Nx,Ny,Nz)

        return self.field

    def field_density_cube(self,pos,d=np.array([1,1,1])):
        """
        Find all the distances in a cube, this method doesn't use any search method but rather indexing into self.grids array
        For every atom, it first finds the nearest index to the atom by simply perform floor(x/dx,y/dy,z/dz). Once the nearest
        index to the atom is found which we will call (ix,iy,iz), it then tries to find all the points in a cube with length 2L 
        centered around the index. L is usually determined by n*sigma. 
        The number of indices that one needs to search in every direction is determined by ceil(L/dx,L/dy,L/dz) which we call (nx,ny,nz). 
        So xrange=(ix-nx,ix+nx), yrange=(iy-ny,iy+ny),zrange=(iz-nz,iz+nz). All the indices can then be found by 
        meshgrid(xrange,yrange,zrange). Then PBC can be easily taken care of by subtracting all indices in the meshgrid that are larger than (nx,ny,nz)
        by (nx,ny,nz) and add (nx,ny,nz) to all the ones that are smaller than 0.
        
        Args:
        ----
            pos(np.ndarray): the positions of the atoms (Ntot,3)
            d(np.ndarray): which dimension will not be ignored (numpy array (3,))

        Return: 
        ------
            a field of shape (Nx,Ny,Nz) from ngrids
        """
        box = self.box
        dbox = self.dbox
        ngrids = self.ngrids
        Nx,Ny,Nz = self.nx,self.ny,self.nz
        sigma = self.sigma

        # create grids and empty field
        grids = self.grids.reshape((Nx,Ny,Nz,3))
        field = np.zeros((Nx,Ny,Nz)) 

        for p in pos: 
            indices = np.ceil(p/dbox)    
            idx = indices + self.ref_idx
            idx %= ngrids
            idx = idx.astype(int)
            
            dr = abs(p - grids[idx[:,0],idx[:,1],idx[:,2]])

            # check pbc
            cond = 1*(dr > box/2)
            cond = cond*d

            # correct pbc
            dr = cond*box - dr
            
            field[idx[:,0],idx[:,1],idx[:,2]] += coarse_grain(dr,sigma)

        return field

    def surface1d(self,field,c=0.016,direction='yz'):
        """
        Function that calculates the points to plot a 1d surface

        Args:
            field(np.ndarray): field passed in with shape(Nx,Ny,Nz)
            c(float): where the isosurface lie (default 0.016 for water)
            direction(str): which direction to average over (default 'yz', sum over y and z)

        Return:
            1.rho1d = numpy array of integrated and averaged density
            2.point = The point where the surface crosses c 
        """ 
        nx,ny,nz = field.shape

        if direction == 'yz':
            n,L = nx,self.Lx
            # sum over y and z
            field = (self.field.sum(axis=-1)/nz).sum(axis=-1)/ny # (Nx,)

        if direction == 'xz':
            n,L = ny,self.Ly
             # sum over x and z
            field = (self.field.sum(axis=0)/nx).sum(axis=-1)/nz # (Ny,)

        if direction == 'xy':
            n,L = nz,self.Lz
            # sum over x and y
            field = (self.field.sum(axis=0)/nx).sum(axis=0)/ny # (Nz,)

        grids = np.linspace(0,L,n) 
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
                interface = pgrid + ratio*(cgrid-pgrid)

            pfield = cfield
            pgrid = cgrid

        return interface, field

    def surface2d(self,field,c=0.016,verbose=False,direction='y'):
        """
        Function that calculates the on the 2d surface from rho(x,z), similar to marching squares algorithm

        Args:
            field(np.ndarray): The density field (Nx,Ny,Nz)
            c(float): where the isosurface lie (default 0.016)
            verbose(bool): whether to be verbose and print things (default False)
            direction(str): which direction to average over for the 2d surface (default 'y')

        Return:    
            A numpy array with points that lies on the 2d isosurface
        """
        nx,ny,nz = field.shape

        if direction == 'y':
            field = field.sum(axis=1)/ny # (Nx,Nz)
            x1 = np.linspace(0,self.Lx,nx)
            x2 = np.linspace(0,self.Lz,nz)

            xx1,xx2 = np.meshgrid(x1,x2)
            grids = np.concatenate((xx1[:,:,np.newaxis],xx2[:,:,np.newaxis]),axis=-1) # (Nx,Nz,2)
            N = nx

        if direction == 'x':
            field = field.sum(axis=0)/nx # (Ny,Nz)
            x1 = np.linspace(0,self.Ly,ny)
            x2 = np.linspace(0,self.Lz,nz)

            xx1,xx2 = np.meshgrid(x1,x2)
            grids = np.concatenate((xx1[:,:,np.newaxis],xx2[:,:,np.newaxis]),axis=-1) # (Nx,Nz,2)
            N = ny

        if direction == 'z':
            field = field.sum(axis=-1)/nz # (Nx,Ny)
            x1 = np.linspace(0,self.Lx,nx)
            x2 = np.linspace(0,self.Ly,ny)

            xx1,xx2 = np.meshgrid(x1,x2)
            grids = np.concatenate((xx1[:,:,np.newaxis],xx2[:,:,np.newaxis]),axis=-1) # (Nx,Nz,2)
            N = nx


        # define "previous field" as in all the field that has the same x coordinate (in this case x=0)
        pfield = field[0]
        # define "previous grids" as in all the grids that has the same x coordinate (in this case x=0)
        pgrids = grids[0]
        points_all = []

        for i in range(1,N):
            # define "current field/grids" as in the current grid/field we are looking at that share the same x coordinate
            cfield = field[i] # (n2, )
            cgrids = grids[i] # (n2,2)
            
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

        surface2d = np.concatenate(points_all)

        # sort surface2d by the second dimension for easier plotting
        order = np.argsort(surface2d[:,-1])
        surface2d = surface2d[order]

        return surface2d
 
    def surface3d(self,field,c=0.016,gradient_direction='descent'):
        """
        Output the vector positions of the triangles needed for graphing isosurface from marching cubes algorithm 

        Args:
            field(np.ndarray): The density field (Nx,Ny,Nz)
            c(float): the contour line value for the isosurface (default 0.016 for water)
            gradient_direction(str): 'descent' if the values exterior of the object are smaller,
                                'ascent' if the values exterior of the object are bigger
        
        Return: 
                the indices for all triangles (N,3,3) where N=number of triangles
        """ 
        dx,dy,dz = self.dx,self.dy,self.dz

        verts,faces,_,_ = measure.marching_cubes_lewiner(field,c,spacing=(dx,dy,dz))

        return verts[faces] 

@njit
def sum_squared_2d_array_along_axis1(arr):
    """
    Function that sums a vector square along axis 1

    Args:
        arr(np.ndarray): A numpy array with shape (N,L) where L is the dimension being summed

    Return:
        A numpy array with shape (N,)
    """
    res = np.empty(arr.shape[0], dtype=arr.dtype)
    for o_idx in range(arr.shape[0]):
        sum_ = 0
        for i_idx in range(arr.shape[1]):
            sum_ += arr[o_idx, i_idx]*arr[o_idx,i_idx] 
        res[o_idx] = sum_
    return res


@jit(nopython=True)
def coarse_grain(dr,sigma):
    """
    coarse graining function for the density of a field

    Args:
        dr(np.ndarray): the vector distance (could be float, 1d np.ndarray vector or 2d np.ndarray matrix)
        sigma(float): the "standard deviation" of the gaussian field applied on each of the molecules
        
    returns:
        the coarse grained density (float, 1d np.ndarray or 2d np.ndarray that matches the input) 
    """
    d = dr.shape[-1]
    sum_ = sum_squared_2d_array_along_axis1(dr)
    sigma2 = np.power(sigma,2)
    
    return np.power(2*np.pi*sigma2,-d/2)*np.exp(-sum_/(2*sigma2))
