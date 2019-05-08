#!/usr/bin/env python

import numpy as np
import filterpicker as FP

fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                      filter_window=200, longterm_window=100, t_up=20,
                      threshold_1=20, threshold_2=10)
pidx, punc, pfrq = fpp.run()
print(pidx, punc, pfrq)
fig = fpp.plot()
