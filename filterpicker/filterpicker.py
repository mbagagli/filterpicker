import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------- Class Def


class FilterPicker(object):
    """
    Implementation of A.Lomax FilterPicker [1] in Python
    Inspiration taken from MATLAB code of Y.Kamer [2]

    *** NB: At the moment the picker works on the entire trace, not
            suitable for online picking.

    *** NB: np.where return a tuple, that's why I use 0
            to retrieve the index array

    REFERENCE:
    [1] Lomax, A., C. Satriano and M. Vassallo (2012), Automatic picker
        developments and optimization: FilterPicker - a robust, broadband
        picker for real-time seismic monitoring and earthquake early-warning,
        Seism. Res. Lett. , 83, 531-540, doi: 10.1785/gssrl.83.3.531.
    [2] www.mathworks.com/matlabcentral/fileexchange/69211-filterpicker-a-robust-broadband-phase-detector-and-picker

    """

    def __init__(self, dt, data,
                 filter_window=3.0,     # dt=0.01 sec --> 300 samples
                 longterm_window=5.0,   # dt=0.01 sec --> 500 samples
                 t_up=0.2,              # dt=0.01 sec --> 500 samples
                 threshold_1=10,
                 threshold_2=10,
                 base=2):
        """
        Initialize the object:
         - dt: is the sampling period of the trace
         - data: is the values
         - filter_window:
         - longterm_window:
        For the picker parameter go throught the references

        """
        self.dt = dt
        self.y = data
        self.tf = self._sec2sample(filter_window, 1.0/self.dt)
        self.tl = self._sec2sample(longterm_window, 1.0/self.dt)
        self.tup = self._sec2sample(t_up, 1.0/self.dt)
        self.thr1 = float(threshold_1)
        self.thr2 = float(threshold_2)
        self.numSMP = len(self.y)
        # MB: next line can be changed to work on smaller portion
        self.veclim = (0, len(self.y))
        self.base = base
        # For new 'get' methods
        self.FnS = None

    def _sec2sample(self, value, df):
        """
        Utility method to define convert USER input parameter (seconds)
        into obspy pickers 'n_sample' units.

        Python3 round(float)==int // Python2 round(float)==float
        BETTER USE: int(round(... to have compatibility

        Formula: int(round( INSEC * df))
        *** NB: input value sec could be float
        """
        return int(round(value * df))

    def _setup(self):
        """ Finalize the parameter preparation based on input """

        # Filter window in dT
        PRM_Tflt = self.base**np.ceil(np.log(self.tf)/np.log(self.base))
        self.PRM_Tlng = self.tl
        self.PRM_Clng = 1 - (1/self.PRM_Tlng)
        self.PRM_Tup = np.max([1, self.tup])
        self.MIN_SIG = np.finfo(float).tiny
        Navg = np.min([np.round(self.PRM_Tlng), self.numSMP])
        self.y0 = np.mean(self.y[0:Navg])
        self.numBnd = int(np.ceil(np.log(PRM_Tflt)/np.log(self.base)))

        self.Fn = np.zeros([self.numBnd, self.numSMP])
        self.FnL = np.zeros([self.numBnd, self.numSMP])
        self.Yout = np.zeros([self.numBnd, self.numSMP])
        return True

    def _loopOverBands(self):
        """ Middle term game of the picker """

        # MB: For each band the CF is created

        for n in range(1, self.numBnd):
            w = (self.base**n*self.dt) / (2*np.pi)
            cHP = w/(w+self.dt)
            cLP = self.dt/(w+self.dt)
            yHP1p = 0
            yHP2p = 0
            yLPp = 0
            #
            avg = 0
            vrn = 0
            sig = 0
            tmpFNL = self.thr1/2

            for i in range(self.veclim[0], self.veclim[1]):
                # First high-pass
                if i != 0:
                    yHP1 = cHP*(yHP1p + self.y[i] - self.y[i-1])
                else:
                    yHP1 = cHP*(yHP1p + self.y[i] - self.y0)
                #
                yHP2 = cHP*(yHP2p + yHP1 - yHP1p)  # Second high-pass
                yLP = yLPp + cLP*(yHP2 - yLPp)     # Low-pass
                En = yLP**2                        # Envelope

                # MB: store anyway for further plots
                self.Yout[n, i] = yLP
                #
                yHP1p = yHP1
                yHP2p = yHP2
                yLPp = yLP
                if (sig > self.MIN_SIG):
                    if ((En-avg) / sig > 5*self.thr1):
                        En = 5*self.thr1 * sig+avg
                        self.Fn[n, i] = 5*self.thr1
                    else:
                        self.Fn[n, i] = (En-avg)/sig  # CF
                        if (self.Fn[n, i] < 1):
                            self.Fn[n, i] = 0.0

                avg = self.PRM_Clng*avg + (1-self.PRM_Clng)*En
                vrn = self.PRM_Clng*vrn + (1-self.PRM_Clng)*(En-avg) ** 2
                sig = np.sqrt(vrn)  # Standard deviation

                tmpFNL = self.PRM_Clng*tmpFNL + (1-self.PRM_Clng)*self.Fn[n, i]
                tmpFNL = min(self.thr1/2, tmpFNL)
                tmpFNL = max(0.5, tmpFNL)
                self.FnL[n, i] = tmpFNL
        #
        self.FnS = self.Fn.max(0)   # Maximum of all bands
        self.FnS[self.FnS < 0] = 0

        convVec = (np.concatenate(([0.0], np.ones(self.PRM_Tup-2), [0.0])) /
                   self.PRM_Tup)
        self.FnMAvg = np.convolve(self.FnS, convVec, 'valid')
        self.PotTrg = np.where(self.FnS > self.thr1)[0]
        return True

    def _analyzeTrigger(self):
        """ Final part of the picker """

        # Next line find index to remove from array
        remidx = np.where((self.PotTrg < (self.PRM_Tlng + self.veclim[0])) |
                          (self.PotTrg > (self.numSMP - self.PRM_Tup))
                          )[0]
        self.PotTrg = np.delete(self.PotTrg, remidx)

        flagTrg = np.full(np.shape(self.PotTrg), True)
        pickIDX = np.full(np.shape(self.PotTrg), np.nan)
        pickUNC = np.full(np.shape(self.PotTrg), np.nan)
        pickFRQ = np.full(np.shape(self.PotTrg), np.nan)

        for i in range(0, len(flagTrg)):  # *** MB check index
            if flagTrg[i]:
                tmpTrg = self.PotTrg[i]
                if self.FnMAvg[tmpTrg] > self.thr2:
                    bandTrg = np.where(self.Fn[:, tmpTrg] > self.thr1)[0][0]

                    firstPot = np.where((self.Fn[bandTrg,
                                                 np.arange(tmpTrg, -1, -1)] -
                                         self.FnL[bandTrg,
                                                  np.arange(tmpTrg, -1, -1)])
                                        < 0)[0][0]
                    #
                    if not firstPot:
                        firstPot = 0

                    pickFRQ[i] = bandTrg
                    pickIDX[i] = (tmpTrg - firstPot)

                    limT = (self.base**bandTrg)/20

                    if firstPot < limT:
                        pickUNC[i] = limT * self.dt
                    else:
                        pickUNC[i] = firstPot * self.dt

                    # Disable next potential picks until FnS drops below 2
                    try:
                        idxDrop = np.where(self.FnS[tmpTrg:] < 2)[0][0]
                    except IndexError:
                        idxDrop = None

                    if idxDrop:
                        flagTrg[(self.PotTrg > tmpTrg) &
                                (self.PotTrg <= (tmpTrg + idxDrop))] = 0
        #
        self.PotTrg = self.PotTrg[~np.isnan(pickIDX)]
        self.pickIDX = pickIDX[~np.isnan(pickIDX)]
        self.pickUNC = pickUNC[~np.isnan(pickUNC)]
        self.pickFRQ = pickFRQ[~np.isnan(pickFRQ)]
        return True

    def _normalizeTrace(self, nparr, rangeVal=[-1, 1]):
        """
        This simple method will normalize the trace between rangeVal.
        Simply by scaling everything...

        OUTPUT: a copy of the array

        *** INPUT MUST BE A numpy.array

        """
        outarr = np.zeros(shape=nparr.shape)
        outarr = (((nparr - np.min(nparr)) / (np.max(nparr) - np.min(nparr))) *
                  (rangeVal[1] - rangeVal[0]))
        outarr = outarr + rangeVal[0]
        return outarr

    # ----------------------------------------------- PUBLIC

    def run(self):
        """
        Orchestrator of the picker, calls in sequence the
        needed functions.

        OUTPUT:
         - pickIDX
         - pickUNC
         - pickFRQ
        """
        self._setup()
        self._loopOverBands()
        self._analyzeTrigger()
        return self.pickIDX*self.dt, self.pickUNC, self.pickFRQ+1

    def plot(self, show=True, fp_param_title=True, fig_title=None):
        """
        Create a comprehensive figure of the different CFs and picks.
        If show=True the fgire is displayed realtime.

        OUTPUT:
            - Return Fig Handle

        """
        # --- Body
        fig, axLst = plt.subplots(nrows=self.numBnd,
                                  sharex=True, sharey=True,
                                  figsize=(10, 7))

        ymax = np.max(self.FnS[self.PRM_Tlng:])
        timeax = np.arange(0, self.numSMP, 1)*self.dt
        pltVec = self._normalizeTrace(self.y, rangeVal=[0, ymax])

        # Raw Plot
        axLst[0].plot(timeax, pltVec[:],
                      '-k', lw=2, label='Raw Input')
        axLst[0].plot(timeax, self.FnS,
                      ':k', lw=1.5, label='max CF')
        axLst[0].plot(np.arange(0, len(self.FnMAvg), 1)*self.dt, self.FnMAvg,
                      ':r', lw=1.5, label='Mov. Avg. (Tup)')
        # 24052019 Next if avoid crashing of software if no pick found
        if self.PotTrg.size and self.pickIDX.size:
            for _xx in range(self.PotTrg.size):
                axLst[0].axvline(self.PotTrg[_xx]*self.dt,
                                 color='b', ls='solid',
                                 lw=2, label='Triggers')
            for _yy in range(self.pickIDX.size):
                axLst[0].axvline(self.pickIDX[_yy]*self.dt,
                                 color='r', ls='solid',
                                 lw=2, label='Picks')
        axLst[0].set_xlim([timeax[0], timeax[-1]])
        axLst[0].legend(loc='upper left')
        #
        for _kk in range(1, self.numBnd):
            # LP
            pltVec = (ymax*self.Yout[_kk, :] /
                      (2*np.max(np.abs(self.Yout[_kk, :]))))
            axLst[_kk].plot(
                        timeax, pltVec, '-',
                        label=('{:.2f}'.format(1/(self.base**(_kk-1)*self.dt))
                               + ' HZ Lp'))

            # CF
            pltVec = self.Fn[_kk, :]
            axLst[_kk].plot(
                        timeax, pltVec, '-', color='gold',
                        label=('{:.2f}'.format(1/(self.base**(_kk-1)*self.dt))
                               + ' HZ Cf'))

            # HZ Accum.
            pltVec = self.FnL[_kk, :]
            axLst[_kk].plot(
                        timeax, pltVec, '-', color='black',
                        label=('{:.2f}'.format(1/(self.base**(_kk-1)*self.dt))
                               + ' HZ Accum.'))

            # Additional
            axLst[_kk].set_xlim([timeax[0], timeax[-1]])
            axLst[_kk].legend(loc='upper left')

        #
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0)
        #
        if fp_param_title:
            fig.suptitle((('TL, TF, Tup, S1, S2' + os.linesep +
                           '%5.2f, %5.2f, %5.2f, %5.2f, %5.2f') %
                         (self.tf, self.tl, self.tup, self.thr1, self.thr2)),
                         fontsize=16, fontweight="bold")
        else:
            if fig_title:
                fig.suptitle(fig_title, fontsize=16, fontweight="bold")

        if show:
            plt.show()
        #
        return fig, axLst

    def get_evaluation_function(self):
        """get_evaluation_function
        """
        if self.FnS:
            return self.FnS
        else:
            raise AttributeError("Missing evaluation function! "
                                 "Use the 'run' method before hand")
