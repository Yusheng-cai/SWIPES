import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from numba import jit,njit
       
class isosurface:
    def __init__(self,box,ngrids,sigma=2.4,n=2.5,kdTree=True,field=None,verbose=False):
        """
        pos: position of the atoms/virtual atoms(COM) in the desired probe volume (N,3),
             these are already normalized where pox[x,y,z] all lie respectively in [0,Lx),[0,Ly),[0,Lz)
        box: a np.ndarray of [Lx,Ly,Lz]
        ngrids: a np.ndarray of [Nx,Ny,Nz]
        sigma: the sigma used for coarse graining of density field
        n: the cutoff radius for n*sigma
        kdtree: whether or not to build kdtree
        verbose: whether or not to print stuff (be verbose)
        """
        self.verbose = verbose
        self.box = box
        self.Lx,self.Ly,self.Lz = box

        self.ngrids = ngrids
        self.nx,self.ny,self.nz = ngrids

        self.dbox = box/ngrids
        self.dx,self.dy,self.dz = self.dbox

        self.sigma = sigma
        self.n = n
        self.L = n*sigma
        self.nidx_search = np.ceil(self.L/self.dbox) 

        # User can pass in a field or else it will be None
        if field is not None:
            if verbose:
                print("You have passed in a density field!")
        self.field = field

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
        self.ref_idx = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

    
    def field_density_kdtree(self,pos,d=np.array([1,1,1])):
        """
        This is not a exact way to find the density field, but cut off the density gaussian at 
        n*sigma. The meshgrid points within the radius are found using kdtree, building of the 
        tree is M log(M) and searching takes log(M). 

        n: the n in the cutoff n*sigma that we want to approximate the density field by
        d: the dimension to be ignored in pbc calculation, a numpy array with shape (3,)

        returns:
            the field density 
        """
        if self.field is not None:
            if self.verbose:
                print("The field that was passed in will now be overwritten")
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

        pos: the positions of the atoms (Ntot,3)
        n: the "radius" of a cube that the code will search for 
        d: which dimension will not be ignored (numpy array (3,))

        returns: 
                a field of shape (Nx,Ny,Nz) from ngrids
        """
        if self.field is not None:
            if self.verbose:
                print("The field that was passed or was just calculated in will now be overwritten") 

        box = self.box
        dbox = self.dbox
        ngrids = self.ngrids
        Nx,Ny,Nz = self.nx,self.ny,self.nz
        sigma = self.sigma

        # create grids and empty field
        grids = self.grids.reshape((Nx,Ny,Nz,3))
        self.field = np.zeros((Nx,Ny,Nz)) 

        for p in pos: 
            indices = np.ceil(p/dbox)    
            idx = self.ref_idx + indices
            idx %= ngrids
            idx = idx.astype(int)

            dr = abs(p - grids[idx[:,0],idx[:,1],idx[:,2]])

            # check pbc
            cond = 1*(dr > box/2)
            cond = cond*d

            # correct pbc
            dr = abs(cond*box - dr)

            self.field[idx[:,0],idx[:,1],idx[:,2]] += coarse_grain(dr,sigma)

        return self.field

    def surface1d(self,field1d=None,grids1d=None,c=0.016,direction='yz'):
        """
        Function that calculates the points to plot a 1d surface

        field1d: the x field usually (Nx,), if None, then will be calculated from self.field by integrating
        out y and z dependence
        grids1d: the x grids usually (Nx,), if None, then will generate grids based on self.Nx and self.Lx
        c: where the isosurface lie
        direction: which direction to average over

        returns:
           The point where the surface crosses c 
        """
        if field1d is None and self.field is None:
            raise RuntimeError("Please provide a field or run self.field_density_cube or self.density_kdtree!")
        
        if direction == 'yz':
            n,L = self.nx,self.Lx
        if direction == 'xz':
            n,L = self.ny,self.Ly
        if direction == 'xy':
            n,L = self.nz,self.Lz

        if field1d is not None:
            field = field1d
        else:
            if direction == 'yz':
                # sum over y and z
                field = (self.field.sum(axis=-1)/self.nz).sum(axis=-1)/self.ny # (Nx,)
            if direction == 'xz':
                # sum over x and z
                field = (self.field.sum(axis=0)/self.nx).sum(axis=-1)/self.nz # (Ny,)
            if direction == 'xy': 
                # sum over x and y
                field = (self.field.sum(axis=0)/self.nx).sum(axis=0)/self.ny # (Nz,)

        if grids1d is None:
            grids = np.linspace(0,L,n)
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
                interface = pgrid + ratio*(cgrid-pgrid)

            pfield = cfield
            pgrid = cgrid

        return interface, field

    def surface2d(self,field2d=None,grids2d=None,c=0.016,verbose=False,direction='y'):
        """
        Function that calculates the points to plot a 2d surface
        fields2d: the x & z field usually (Nx,Nz), if None, then will use self.field
        grids2d: the x & z grids usually, if None, then will generate grids
        c: where the isosurface lie
        verbose: whether to be verbose and print things
        direction: which direction to average over for the 2d surface

        returns:    
            a list of points that lies on the 2d isosurface
        """
        if field2d is None and self.field is None:
            raise RuntimeError("Please provide a field or run self.field_density_cube or self.density_kdtree!")
        
        if field2d is not None:
            field = field2d
        else:
            if direction == 'y':
                field = self.field.sum(axis=1)/self.ny # (Nx,Nz)
            if direction == 'x':
                field = self.field.sum(axis=0)/self.nx # (Ny,Nz)
            if direction == 'z':
                field = self.field.sum(axis=-1)/self.nz # (Nx,Ny)

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

        surface2d = np.concatenate(points_all)
        order = np.argsort(surface2d[:,-1])
        surface2d = surface2d[order]

        return surface2d

 
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

@njit
def sum_squared_2d_array_along_axis1(arr):
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
    dr: the vector distance (could be float, 1d np.ndarray vector or 2d np.ndarray matrix)
    sigma: the "standard deviation" of the gaussian field applied on each of the molecules
    
    returns:
        the coarse grained density (float, 1d np.ndarray or 2d np.ndarray that matches the input) 
    """
    d = dr.shape[-1]
    sum_ = sum_squared_2d_array_along_axis1(dr)
    sigma2 = np.power(sigma,2)
    
    return np.power(2*np.pi*sigma2,-d/2)*np.exp(-sum_/(2*sigma2))
