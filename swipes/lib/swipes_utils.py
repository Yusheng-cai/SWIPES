import numpy as np

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
