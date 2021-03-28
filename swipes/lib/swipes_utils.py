import MDAnalysis as mda
import numpy as np
from scipy.special import erf

def h(alpha_i,alpha_c,sigma,amin,amax):
    """
    This is the function h(alpha) used in the paper by Amish on INDUS
    This function that the form 

    h(alpha_i) = \int_{amin}^{amax} \Phi(alpha-alpha_i) dr
    where 

    \phi(alpha_i) = k^-1*[e^{-alpha^{2}/(2sigma^{2})} - e^{-alphac^{2}/(2sigma^{2})}]
    where k is the normalizing constant
    
    Args:
        alpha_i: the input alpha's, it can be within range range (can be a numpy array or float or int)
        alpha_c: the alpha_c in the equation
        sigma: the sigma in the equation
        amin: the lower bound of integral
        amax: the upper bound of integral

    Return
        a float/int or numpy array depending on the input alpha_i
        if alpha_i is float/int, then output will be int that corresponds to h(alpha_i)
        else if alpha_i is numpy array, then output will be numpy array that corresponds to h(alpha_i)
    """
    # normalizing constant
    k = -2*np.exp(-alpha_c**2/(2*sigma**2))*alpha_c+np.sqrt(2*np.pi*sigma**2)*erf(alpha_c/np.sqrt(2*sigma**2))

    # the low and high of the function, beyond these will be zero
    low = amin - alpha_c
    high = amax + alpha_c
    h = np.heaviside(alpha_i - low,1) - np.heaviside(alpha_i - high,1)
    
    # set appropriate boundary depending on the alpha_i
    a = np.where(np.abs(alpha_i - amin) < alpha_c,amin,alpha_i-alpha_c)
    b = np.where(np.abs(alpha_i - amax) < alpha_c,amax,alpha_i+alpha_c)

    # return the integrated value/values
    return h/k*((a-b)*np.exp(-alpha_c**2/(2*sigma**2))+\
            np.sqrt(np.pi/2)*sigma*(erf((alpha_i-a)/np.sqrt(2*sigma**2))\
                                    -erf((alpha_i-b)/np.sqrt(2*sigma**2)))) 

def Ntilde(pos,xmin,xmax,ymin,ymax,zmin,zmax,sigma=0.01,alpha_c=0.02):
    """
    Calculating Ntilde in a probe volume spanned by {xmin,xmax},{ymin,ymax},{zmin,zmax}
    h(r) = h(x)h(y)h(z)

    Args:
        pos(numpy.ndarray): The positions of all the particles 
        xmin(float): The minimum x of the probe volume
        xmax(float): The maximum x of the probe volume
        ymin(float): The minimum y of the probe volume
        ymax(float): The maximum y of the probe volume
        zmin(float): The minimum z of the probe volume
        zmax(float): The maximum z of the prbe volume
        sigma(float): The sigma parameters in the coarse-grained function
        alpha_c(float): The alpha_c parameter in the coarse-grained function.

    Return:
        Ntilde(float): The Ntilde of the system
    """
    hx_ = h(pos[:,0],alpha_c,sigma,xmin,xmax)
    hy_ = h(pos[:,1],alpha_c,sigma,ymin,ymax)
    hz_ = h(pos[:,2],alpha_c,sigma,zmin,zmax)

    h_ = hx_*hy_*hz_
    Ntilde = h_.sum()

    return Ntilde

def cubic_lattice(Sx,Sy,Sz,lattice_spacing):
    """
    Function that create a cubic lattice, the bottom left edge will always be at [0,0,0]

    Args:
        Sx(float): The length of the box in the x direction (in nm)
        Sy(float): The length of the box in the y direction (in nm)
        Sz(float): The length of the box in the z direction (in nm)
        lattice_spacing(float): The lattice spacing between LJ atoms (in nm)

    Return: 
        coordinates for every atom in the lattice (Ntot,3)
    """
    if Sx % lattice_spacing == 0:
        Sx += lattice_spacing
    if Sy % lattice_spacing == 0:
        Sy += lattice_spacing
    if Sz % lattice_spacing == 0:
        Sz += lattice_spacing

    x = np.arange(0,Sx,lattice_spacing)
    y = np.arange(0,Sy,lattice_spacing)
    z = np.arange(0,Sz,lattice_spacing)
    xx,yy,zz = np.meshgrid(x,y,z)
    coords = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

    return coords


def write_LJparticle_gro(coords,atom_name='WALL',atom_type='Y',file_path='WALL.gro'):
    N = len(coords)
    first_line = "Simulation of {} LJ particles\n".format(N)
    second_line = "{}\n".format(N)
    gromacs_format = "{0:>5d}{1:<5s}{2:>5s}{3:>5d}{4:>8.3f}{5:>8.3f}{6:>8.3f}\n"
    Sx = np.max(coords[:,0])
    Sy = np.max(coords[:,1])
    Sz = np.max(coords[:,2])
    
    f = open(file_path,"w")
    f.write(first_line)
    f.write(second_line)
    for i in range(N):
        x = coords[i,0]
        y = coords[i,1]
        z = coords[i,2]
        index = i + 1
        f.write(gromacs_format.format(i+1,atom_name,atom_type,index,x,y,z))
    f.write("\t{}\t{}\t{}\n".format(Sx,Sy,Sz))


def findnumparticles(u,res_name,tmin,tmax,xmin=-np.inf,xmax=np.inf,ymin=-np.inf,ymax=np.inf,zmin=-np.inf,zmax=np.inf,alpha_c=0.02,sigma=0.01,skip=1,verbose=False):
    """
    Function that finds the number of particles in a particular probe volume

    Args:
        u(mda.Universe): a MDAnalysis Universe object
        tmin(int): start of the time index at which the calculation is performed
        tmax(int): end of the time index at which the calculation is performed
        res_name(string): The residue that we are finding the numbers for
        xmin(float): start of x of the probe volume (default -np.inf)
        xmax(float): end of x of the probe volume (default np.inf)
        ymin(float): start of y of the probe volume (default -np.inf)
        ymax(float): end of y of the probe volume (default np.inf)
        zmin(float): start of z of the probe volume (default -np.inf)
        zmax(float): end of z of the probe volume (default np.inf)
        alpha_c(float): The alpha_c value in INDUS coarse grain function (cut-off of Gaussian) (default 0.02)
        sigma(float): The standard deviation of the INDUS coarse grain function (cut-off of Gaussian) (detaul 0.01)
        skip(int): The number of frames to skip form tmin to tmax (default 1)

    Returns:
        num_particles(numpy.ndarray): A numpy array of number of particles per time frame 
    """
    time_range = np.arange(tmin,tmax,step=skip)
    T = len(time_range)
    num_particles = np.zeros((T,2))
    ix = 0

    for t in time_range:
        u.trajectory[t]
        pos = u.select_atoms(res_name).positions

        pos = pos[pos[:,0] >= xmin]
        pos = pos[pos[:,0] <= xmax]

        pos = pos[pos[:,1] >= ymin]
        pos = pos[pos[:,1] <= ymax]

        pos = pos[pos[:,2] >= zmin]
        pos = pos[pos[:,2] <= zmax]
        

        num_particles[ix,0] = len(pos)
        num_particles[ix,1] = Ntilde(pos,xmin,xmax,ymin,ymax,zmin,zmax,sigma=sigma,alpha_c=alpha_c)
        if verbose:
            print("{} is done".format(t))
        ix += 1

    return num_particles
