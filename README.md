# SWIPES
SWIPES is a method developed at Patel group at the University of Pennsylvania to find contact angles of liquid droplets from Molecular Dynamics simulation. For detail information, please refer to Jiang, H., Fialoke, S., Vicars, Z. & Patel, A. J. Characterizing surface wetting and interfacial properties using enhanced sampling (SWIPES). Soft Matter 15, 860–869 (2019). A crucial part of the code is also to find isosurface, for that please refer to the Willard & Chandler paper of finding instantaneous isosurface at Willard, A. P. & Chandler, D. Instantaneous liquid interfaces. J. Phys. Chem. B 114, 1954–1958 (2010).
The documentation for th code is at: https://yusheng-cai.github.io/SWIPES/

## Installation
Download the source code and run 

```bash
python setup.py install
```

## Usage

```python
from swipes.isosurface import *
import numpy as np

iso = isosurface()
```

## Test

In swipes/test/ folder run the following command 

```bash
pytest 
```



## License
[MIT](https://choosealicense.com/licenses/mit/)
