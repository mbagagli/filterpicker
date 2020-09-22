# Filter Picker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3609025.svg)](https://doi.org/10.5281/zenodo.3609025)

AUTHOR: _Matteo Bagagli_
VERSION: _1.0.5_
DATE: _09/2020_

FilterPicker is a general purpose, broad-band, phase detector and picker which is applicable to real-time seismic monitoring and earthquake early-warning.
This module is a Python implementation of the A.Lomax Filter Picker. Inspiration has been taken from the MATLAB [implementation](https://ch.mathworks.com/matlabcentral/fileexchange/69211-filterpicker-a-robust-broadband-phase-detector-and-picker)[3] of Y.Kamer. This picker class has been rewritten using NumPy libraries.
For a full method description the reader is referred to the main authors paper [1][2]

If you make use of this package, please **consider citing** it with the provided DOI. Thanks :)

```
@misc{mbagagli_2019_3609025,
    author       = {Matteo Bagagli},
    title        = {filterpicker: general purpose, broad-band, phase detector and picker},
    month        = Nov,
    year         = 2019,
    doi          = {10.5281/zenodo.3609025},
    version      = {1.0.3},
    publisher    = {Zenodo},
    url          = {https://doi.org/10.5281/zenodo.3609025}
    }
```

## Installation

Recently the package has been uploaded in _PyPI_ repository, therefore you could just type:
```
pip install filterpicker
```
and be ready to go.

Please note that this package has been **fully tested with Python 3.6 and Python 3.7 interpreters**. Other Python3 versions support is ongoing.

If you want to install the library manually or just being updated to the latest patches, the installation is pretty easy because the package comes with an installer. All the dependencies are explained in the `requirements.txt` file. It's suggested to use a virtual environment (`conda` or `pipenv`)

Just open a terminal and type
```
$ git clone https://github.com/billy4all/filterpicker /somwhere/in/my/pc
$ cd where/you/cloned
$ # optionally activate your virtual env
$ pip install .
```

## Tests

To run a simple test to make sure you're ready to go, just type:
```
$ cd where/the/package/is
$ pytest
```

You can also double check the performance of the software by running the scripts in the `example` folder (manual installation) or run the command-line exec store `obspy_script` (PyPI).
The module is fully compatible with the widely used **ObsPy** library: just fed the picker with the trace data (`obspy.core.Trace.data` numpy array). It will work with any numpy.array, though.

For any issues/bug reports, please send an email to: _matteo.bagagli@erdw.ethz.ch_
Enjoy ^-^

##### References

[1] Lomax, A., C. Satriano and M. Vassallo (2012), Automatic picker developments and optimization: FilterPicker - a robust, broadband picker for real-time seismic monitoring and earthquake early-warning, Seism. Res. Lett. , 83, 531-540, doi: 10.1785/gssrl.83.3.531.

[2] Vassallo, M., C. Satriano and A. Lomax, (2012), Automatic picker developments and optimization: A strategy for improving the performances of automatic phase pickers, Seism. Res. Lett. , 83, 541-554, doi: 10.1785/gssrl.83.3.541.

[3] MATLAB packagehttps://ch.mathworks.com/matlabcentral/fileexchange/69211-filterpicker-a-robust-broadband-phase-detector-and-picker
