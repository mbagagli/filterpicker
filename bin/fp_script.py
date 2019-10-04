#!/usr/bin/env python

import numpy as np

from filterpicker import filterpicker as FP

fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                      filter_window=1.6, longterm_window=3.2, t_up=0.16,
                      threshold_1=20, threshold_2=10)
pidx, punc, pfrq = fpp.run()
print(pidx, punc, pfrq)
fig = fpp.plot()
