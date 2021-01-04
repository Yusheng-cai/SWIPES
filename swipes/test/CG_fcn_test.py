from swipes.isosurface import coarse_grain,sum_squared_2d_array_along_axis1
import numpy as np
import timeit
from numba import jit,njit

def c(dr,sigma):
    d = dr.shape[-1]
    sum_ = (dr**2).sum(axis=-1)
    sigma2 = np.power(sigma,2)
    
    return np.power(2*np.pi*sigma2,-d/2)*np.exp(-sum_/(2*sigma2))

if __name__ == '__main__':
    SETUP_CODE = '''
from swipes.isosurface import coarse_grain,sum_squared_2d_array_along_axis1
import numpy as np  
    '''

    TEST_CODE = '''
dr = np.random.randn(100000,3)
coarse_grain(dr,3)
    '''

    print(timeit.timeit(setup=SETUP_CODE,stmt=TEST_CODE,number=50))
