# Filter Picker
=========================

*  AUTHOR: _Matteo Bagagli_
* VERSION: _1.0.1_
*    DATE: _06/2019_

FilterPicker is a general purpose, broad-band, phase detector and picker which is applicable to real-time seismic monitoring and earthquake early-warning.
Python implementation of the A.Lomax Filter Picker. Inspiration has been taken from the MATLAB [implementation](https://ch.mathworks.com/matlabcentral/fileexchange/69211-filterpicker-a-robust-broadband-phase-detector-and-picker)[3] of Y.Kamer. This picker class has been rewritten using NumPy libraries.

For a full reference of usage/method the reader is referenced to the main author paper [1][2]

## Installation


Recently the package has been uploaded in _PyPi_ repository, you could just type:
```
pip install filterpicker
```

Otherwise you could install the library manually. The installation is pretty easy because the package comes with an installer. All the dependencies are explained in the `requirements.txt` file. It's suggested to use a virtual environment (`conda` or `pipenv`)

```
$ cd where/you/cloned
$ # optionally activate your virtual env
$ pip install .
$ pytest
```

## Tests
To run the simple test you just need to type:
```
$ cd where/the/package/is
$ pytest
```

If all test are passed you can verify the installation by running:
```
import numpy as np
from filterpicker import filterpicker as FP
fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                      filter_window=1.6, longterm_window=3.2, t_up=0.16,
                      threshold_1=20, threshold_2=10)
pidx, punc, pfrq = fpp.run()
print(pidx, punc, pfrq)
fig = fpp.plot()
```

The previous set of commands are already stored in a script `./bin/fp_script.sh`


Enjoy ^-^

##### References
-------------------------
[1] Lomax, A., C. Satriano and M. Vassallo (2012), Automatic picker developments and optimization: FilterPicker - a robust, broadband picker for real-time seismic monitoring and earthquake early-warning, Seism. Res. Lett. , 83, 531-540, doi: 10.1785/gssrl.83.3.531.

[2] Vassallo, M., C. Satriano and A. Lomax, (2012), Automatic picker developments and optimization: A strategy for improving the performances of automatic phase pickers, Seism. Res. Lett. , 83, 541-554, doi: 10.1785/gssrl.83.3.541.

[3] MATLAB packagehttps://ch.mathworks.com/matlabcentral/fileexchange/69211-filterpicker-a-robust-broadband-phase-detector-and-picker
