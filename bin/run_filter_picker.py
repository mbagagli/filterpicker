#!/usr/bin/env python

import sys, os
import argparse
import logging
from pathlib import Path
from filterpicker import __version__, __author__, __date__
from filterpicker import filterpicker as FP


#  filter_window(s) longterm_window(s)  t_up  threshold_1  threshold_2  base
fp_par_defaults = [0.20, 1.0, 0.1, 5, 10, 2]


# =================================================================== #
#                           Parser SETUP
# =================================================================== #
parser = argparse.ArgumentParser(
                description=(
                    "Script to run the filterpicker program from command line."
                    " It internally uses the Python package of filterpicker "
                    "and provides a useful handle for extracting infos"
                    "User may specify a path to a valid ObsPy Stream format "
                    "strored on disk. If multiple traces in Stream, all of them "
                    "will be picked. The `delta/dt` information is automatically "
                    "extracted from the trace object itself."
                    "Please read this helper for a better flag-usage and "
                    "understanding. If multiple entry per-flag are required, "
                    "use the space as a delimiter between parameters."
                    ))
parser.add_argument("inputfile", type=str,
                    help="Path to a valid seismic-format that can be handled"
                    "by obspy.")
parser.add_argument("-p", "--parameters", type=float, nargs="+", default=None,
                    help=("Parameter list for filterpicker program: `filter_window` "
                          "`longterm_window` `t_up` `threshold_1` `threshold_2` "
                          "`base`. For details on usage, please refer to the "
                          "rreferences listed in the README.md file."))
parser.add_argument("-o", "--outputfile", type=str, default=None,
                    help="output filename (path) of the filterpicker CSV file.")
parser.add_argument("-x", "--processtrace", type=str, default=None, nargs="+",
                    help="If specified, the input trace(s) will be pre-processed "
                    "prior the picking stage. With this flag, the user may "
                    "specify the filter-type, frequency and n-poles to apply to the "
                    "traces. A detrend and mean removal stages are always applied "
                    "prior the filtering. [ type npoles f1 f2]")
parser.add_argument("--plot", action="store_true", dest="doplot",
                    help="It will create a debug-plot for each trace, after picking "
                         "stage. This is useful for debug and quick-checks.")

parser.add_argument("-v", "--version", action="version",
                    version="%(prog)s " + __version__)

# ----------- Set Defaults
parser.set_defaults(parameters=fp_par_defaults)
parser.set_defaults(outputfile="fp_picks.csv")

# ----------- Parse IT
args = parser.parse_args()


# =================================================================== #
#                           Setup Logger
# =================================================================== #

FMT = "[{levelname:^9}] {message}"
FORMATS = {
    logging.DEBUG:     f"\33[36m{FMT}\33[0m",
    logging.INFO:      FMT,
    logging.WARNING:   f"\33[33m{FMT}\33[0m",
    logging.ERROR:     f"\33[31m{FMT}\33[0m",
    logging.CRITICAL:  f"\33[1m\33[31m{FMT}\33[0m"
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt, style="{")  # needed for custom styling
        return formatter.format(record)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    )
logger = logging.getLogger(__name__)


# =================================================================== #
#                           Functions
# =================================================================== #

def _miniproc(prs, proc_par):
    """Private function for processing.

    Args:
        - tt (str): type name of the filter
        - fl (list/tuple): float list of frequencies
        - np (int): number of poles

    Returns:
        - prs (obspy.Trace): processed stream

    """

    tt, np, fl = proc_par[0], proc_par[1], proc_par[2:]

    if len(fl) == 1 and tt.lower() in ("bp", "bandpass"):
        raise ValueError("I need 2 frequency limit for bandpass filtering!")

    prs = st.copy()
    prs.detrend("simple")
    prs.detrend("demean")
    # prs.taper(max_percentage=0.05, type="cosine")

    if tt.lower() in ("bp", "bandpass"):
        prs.filter("bandpass",
                   freqmin=float(fl[0]),
                   freqmax=float(fl[1]),
                   corners=int(np),
                   zerophase=True)
    elif tt.lower() in ("hp", "highpass"):
        prs.filter("highpass",
                   freq=float(fl[0]),
                   corners=int(np),
                   zerophase=True)
    elif tt.lower() in ("lp", "lowpass"):
        prs.filter("lowpass",
                   freq=float(fl[0]),
                   corners=int(np),
                   zerophase=True)
    else:
        raise ValueError("Unknown filter type: %r" % tt)
    #
    return prs


# =================================================================== #
#                           Body
# =================================================================== #

# 0)  Check/unpack parameters
try:
    import obspy
except ImportError:
    logger.error("ObsPy library missing! In order to work, this script needs it."
                 " Please install it by tiping `pip install obspy`.")
    sys.exit()

if not args.inputfile or not Path(args.inputfile).exists():
    logger.error("Input file is missing or with a wrong path!")
    sys.exit()

logger.info("Working with:  %s" % args.inputfile)
logger.info("Filter Picker parameters: %r" % args.parameters)


# 1) Load and Pre-process
st = obspy.read(args.inputfile)

if args.processtrace:
    logger.info("Processing input-stream: %r" % args.processtrace)
    st = _miniproc(st, args.processtrace)

# 2) Pick + Store CSV
logger.info("Start Picking Time:  %s" % obspy.UTCDateTime().datetime)
with open(args.outputfile, "w") as OUT:
    OUT.write("TRACE_ID, DF, RELATIVE_SEC, PICK_UTC, PICK_ERR, PICK_BAND"+os.linesep)
    for tr in st:
        myfp = FP.FilterPicker(tr.stats.delta, tr.data, *args.parameters)
        pickTime_relative, pickUnc, pickBand = myfp.run()
        # ----------------------------------------------------
        if (len(pickTime_relative) != len(pickUnc) or
           len(pickTime_relative) != len(pickBand)):
            logger.critical("FATAL: FilterPicker results must be of the same length! "
                            "If you can reproduce the error, open an issue on the project "
                            "GitHub project page")
            sys.exit()
        elif not pickTime_relative:
            logger.warning("No picks found for trace: %s" % tr.id)
        # ----------------------------------------------------
        # Write out
        for xx in range(0, len(pickTime_relative)):
            OUT.write(("%15s, %5.1f, %8.3f, %s, %4.2f, %4.1f" + os.linesep) %
                (tr.id, tr.stats.sampling_rate,
                 pickTime_relative[xx],
                 (tr.stats.starttime + pickTime_relative[xx]).datetime,
                 pickUnc[xx], pickBand[xx])
                )
        # ----------------------------------------------------
        # Plot Interactive
        if args.doplot:
            myfp.plot()

logger.info("End   Picking Time:  %s" % obspy.UTCDateTime().datetime)
