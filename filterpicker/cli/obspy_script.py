#!/usr/bin/env python

# This example follow the guidelines explained at:
# https://docs.obspy.org/tutorial/code_snippets/trigger_tutorial.html

from obspy.core import read
from filterpicker import filterpicker as FP


def main():
    # Read the trace
    st = read("https://examples.obspy.org/ev0_6.a01.gse2")
    st = st.select(component="Z")
    tr = st[0]

    # Apply the picker
    fpp = FP.FilterPicker(0.008, tr.data,
                          filter_window=1.6, longterm_window=3.2, t_up=0.16,
                          threshold_1=20, threshold_2=10)
    pidx, punc, pfrq = fpp.run()

    # Display results
    print(pidx, punc, pfrq)
    fpp.plot()

if __name__ == '__main__':
    main()
