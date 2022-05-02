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





# def _export_fp(pickTime_relative, pickUnc, pickBand, ot):
#     """ Export a CSV and returns a pd.DataFrame """
#     def _calc_abs(row, ot):
#         _at = ot + row["RelativeSeconds"]
#         return _at.datetime

#     data = {"Phase": "FP",
#             "RelativeSeconds": pickTime_relative,
#             "Uncertainty": pickUnc,
#             "TrigBand": pickBand,
#             }
#     outdf = pd.DataFrame(data)
#     outdf["AbsolutePickTime"] = outdf.apply (lambda row: _calc_abs(row, ot), axis=1)
#     return outdf


# def run_fp(tr, fp_par=fp_par, export=True):
#     print("Running ... FilterPicker")
#     myfp = FP.FilterPicker(tr.stats.delta, tr.data, **fp_par)
#     pickTime_relative, pickUnc, pickBand = myfp.run()  # all of results are lists
#     pdout = _export_fp(pickTime_relative, pickUnc, pickBand, tr.stats.starttime)
#     return myfp, pdout


# def run_host(tr, time_win=host_time_win):
#     print("Running ... HOST")
#     hostobj = Host(tr, time_win)
#     kurt_dict = {}
#     skew_dict = {}
#     #
#     print("kurt")
#     hostobj.set_hos_method("kurt")
#     for ii in time_win:
#         kurt_dict["KURT: %.2f" % ii] = hostobj.calculate_single_hos(
#                                             ii, shift_origin=False)
#     #
#     print("skew")
#     hostobj.set_hos_method("skew")
#     for ii in time_win:
#         kurt_dict["SKEW: %.2f" % ii] = hostobj.calculate_single_hos(
#                                             ii, shift_origin=False)
#     #
#     return hostobj, skew_dict, kurt_dict


# def go_pick_yourself(inst, channel="*Z", fp_par=fp_par, host_time_win=host_time_win):
#     outst = Stream()

#     # --- Load
#     intr = inst.select(channel="*Z")[0]

#     # --- FilterPicker
#     fpobj, fppd = run_fp(intr, fp_par=fp_par, export=True)
#     fp_big_cf = fpobj.get_evaluation_function()
#     fp_single_cf = fpobj.get_bands()
#     fpobj.plot()

#     # --- Compare picks
#     delta_match = []
#     def _make_datetime_arr(row, key):
#         return UTCDateTime(row[key])

#     # Manual
#     csv_man = pd.read_csv("man.csv")
#     csv_man["DateTimeVal"] = csv_man.apply (lambda row: _make_datetime_arr(row, "Time"), axis=1)
#     csv_man = csv_man.sort_values(by="DateTimeVal")
#     man_pser = csv_man.loc[csv_man["Phase"] == "P", "DateTimeVal"]
#     # man_pser = csv_man.loc[csv_man["Phase"] == "S", "DateTimeVal"]
#     man_plst = man_pser.to_list()

#     # FP
#     fppd.to_csv(
#         "manVSmachine_FP.csv",
#         sep=",",
#         index=False,
#         na_rep="NA", encoding="utf-8")
#     fppd["DateTimeVal"] = fppd.apply (lambda row: _make_datetime_arr(row, "AbsolutePickTime"), axis=1)
#     fppd = fppd.sort_values(by="DateTimeVal")
#     fp_pser = fppd["DateTimeVal"]
#     fp_plst = fp_pser.to_list()

#     for mn in man_plst:
#         for au in fp_plst:
#             _sec = au - mn
#             if np.abs(_sec) <= 0.2:   # MATCH!
#                 delta_match.append(_sec)
#     #
#     print("Total of Manual List: %d" % len(man_plst))
#     print("Total of Automatic List: %d" % len(fp_plst))
#     print("Total Match %d" % len(delta_match))

#     # Plot
#     # binwidth = 0.01
#     # plt.hist(delta_match, bins=np.arange(min(delta_match), max(delta_match) + binwidth, binwidth))
#     # plt.show()

#     # --- Host
#     hostobj, skew_dict, kurt_dict = run_host(intr, time_win=host_time_win)

#     # --- Append and close
#     print("")
#     print("Appending traces to Stream ...")

#     xx = 0
#     print("  Trace %2d: FP --> Final CF" % xx)
#     outst.append(Trace(data=fp_big_cf))
#     print("      %d" % len(fp_big_cf))
#     xx += 1

#     for kk, dd in fp_single_cf.items():
#         print("  Trace %2d:  FP --> %s" % (xx, kk))
#         outst.append(Trace(data=dd))
#         print("      %d" % len(dd))
#         xx += 1

#     for kk, dd in kurt_dict.items():
#         print("  Trace %2d:  KURT --> %s" % (xx, kk))
#         outst.append(Trace(data=dd))
#         print("      %d" % len(dd))
#         xx += 1

#     for kk, dd in skew_dict.items():
#         print("  Trace %2d:  SKEW --> %s" % (xx, kk))
#         outst.append(Trace(data=dd))
#         print("      %d" % len(dd))
#         xx += 1

#     print("")
#     print("Storing mseed")
#     outst.write("mlcfs.mseed")


# if __name__ == "__main__":
#     st = obspy.read("crcl.mseed")
#     st = miniproc(st)
#     go_pick_yourself(st, channel="*Z")

# # Running ... FilterPicker
# # Total of Manual List: 77
# # Total of Automatic List: 82
# # plotting 66
