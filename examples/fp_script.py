#!/usr/bin/env python

import os
import numpy as np
from filterpicker import filterpicker as FP

moduledir = os.path.dirname(FP.__file__)
fpp = FP.FilterPicker(0.008, np.loadtxt(moduledir + '/tests/fg_sac.npa'),
                      filter_window=1.6, longterm_window=3.2, t_up=0.16,
                      threshold_1=20, threshold_2=10)
pidx, punc, pfrq = fpp.run()
print(pidx, punc, pfrq)
fpp.plot()
