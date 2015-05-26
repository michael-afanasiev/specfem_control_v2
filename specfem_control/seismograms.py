#!/usr/bin/env python

import os
import math
import utils
import obspy

import numpy as np
import matplotlib.pyplot as plt

from lxml import etree

class SeismogramNotFoundError(Exception):
    pass


def plot_two(s1, s2, process_s1=False, process_s2=True, plot=True, ax=None,
             legend=True, xlabel=None, ylabel=None, third=None, window=None):
    
    if process_s1:
        s1.process_synthetics()
    if process_s2:
        s2.process_synthetics()
    if third:
        third.process_synthetics()
    
    norm_data_s1 = s1.normalize()
    time_s1 = s1.t / 60.
    
    norm_data_s2 = s2.normalize()
    time_s2 = s2.t / 60.

    if third:
        norm_data_s3 = third.normalize()
        time_s3 = third.t / 60.
                
    ymax = max(np.amax(np.absolute(norm_data_s1)), 
               np.amax(np.absolute(norm_data_s2))) * (1.2)
    ymin = max(np.amax(np.absolute(norm_data_s1)), 
               np.amax(np.absolute(norm_data_s2))) * (-1.2)            
        
    
    print s1.fname
    print s2.fname

    print s1.tr.stats.starttime
    
    if ax == None:
        ax = plt.gca()
        ax.set_xlabel('Time (m)')
        ax.set_ylabel('Amplitude (normalized)')
    ax.set_xlim(0, max(time_s1))
    ax.set_ylim(ymin, ymax)
    
    if window:
        windows = []
        root = etree.parse(window).getroot()
        for element in root:
            if element.tag == 'Window':
                w = {}
                for elem in element:
                    if elem.tag == 'Starttime':
                        w['starttime'] = (obspy.UTCDateTime(elem.text) - s1.tr.stats.starttime) / 60.
                    if elem.tag == 'Endtime':
                        w['endtime'] = (obspy.UTCDateTime(elem.text) - s1.tr.stats.starttime) / 60.
                        
                windows.append(w)
    
    # Plot datas.
    ax.plot(time_s1, norm_data_s1, 'k', label='Data')#s1.fname)
    ax.plot(time_s2, norm_data_s2, 'r', label='00_globe')#s2.fname)
    if third:
        ax.plot(time_s3, norm_data_s3, 'b', linestyle='-', label='02_globe')#third.fname)
    if window:
        for w in windows:
            ax.axvline(x=w['starttime'])
            ax.axvline(x=w['endtime'])

    legend = True
    if legend:    
        ax.legend(prop={'size':6})

    if plot:
        plt.show()
        plt.savefig('seismogram_00_globe.png')
        
    
    
class Seismogram(object):

    def __init__(self, file_name, net=None, sta=None, cmp=None):

        if file_name.endswith('.ascii'):
            # Read in seismogram.
            temp = np.loadtxt(file_name)
            self.t, self.data = temp[:, 0], temp[:, 1]
            self.fname = file_name
            self.directory = os.path.dirname(file_name)

            # Initialize obspy
            self.tr = obspy.Trace(data=self.data)
            self.tr.stats.delta = (self.t[1] - self.t[0])
            self.tr.stats.sampling_rate = 1 / self.tr.stats.delta
            self.tr.stats.network, self.tr.stats.station, \
                self.tr.stats.channel = \
                os.path.basename(self.fname).split('.')[:3]
            self.tr.stats.channel = self.tr.stats.channel[2]

            # Reverse X component to agree with LASIF
            if self.tr.stats.channel == 'X':
                self.data = self.data * (-1)
                
        elif file_name.endswith('.mseed') or file_name.endswith('.sac'):
            if not net and not sta and not cmp:        
                self.tr = obspy.read(file_name)[0]
            else:
                st = obspy.read(file_name)
                print st.__str__(extended=True)
                for self.tr in st:
                    if sta in self.tr.stats.station and net in self.tr.stats.network and \
                        cmp in self.tr.stats.channel:
                        print sta
                        break
            self.fname = "%s.%s.%s" % (self.tr.stats.network, self.tr.stats.station, self.tr.stats.channel)
            self.t = np.array(
                [x*self.tr.stats.delta for x in range(0, self.tr.stats.npts)])
        else:
            raise SeismogramNotFoundError(utils.print_red("Seismogram not "
                "found."))
                
        
    def process_synthetics(self): 
        
        lowpass_freq = 1/60.
        highpass_freq = 1/120.
    
        freqmin=highpass_freq
        freqmax=lowpass_freq

        f2 = highpass_freq
        f3 = lowpass_freq
        f1 = 0.8 * f2
        f4 = 1.2 * f3
        pre_filt = (f1, f2, f3, f4)
    
        self.tr.differentiate()
    
        # self.tr.data = convolve_stf(self.tr.data)

        self.tr.detrend("linear")
        self.tr.detrend("demean")
        self.tr.taper(max_percentage=0.05, type="hann")

        # Perform a frequency domain taper like during the response removal
        # just without an actual response...

        data = self.tr.data.astype(np.float64)
        orig_len = len(data)

        # smart calculation of nfft dodging large primes
        from obspy.signal.util import _npts2nfft
        from obspy.signal.invsim import c_sac_taper
        nfft = _npts2nfft(len(data))

        fy = 1.0 / (self.tr.stats.delta * 2.0)
        freqs = np.linspace(0, fy, nfft // 2 + 1)

        # Transform data to Frequency domain
        data = np.fft.rfft(data, n=nfft)
        data *= c_sac_taper(freqs, flimit=pre_filt)
        data[-1] = abs(data[-1]) + 0.0j
        # transform data back into the time domain
        data = np.fft.irfft(data)[0:orig_len]
        # assign processed data and store processing information
        self.tr.data = data

    def normalize(self):
        """
        Returns a normalized seismogram.
        """
        max_val = np.amax(np.absolute(self.tr.data))
        return self.tr.data / max_val

    def convert_to_velocity(self):
        """
    Uses a centered finite-difference approximation to convert a
        displacement seismogram to a velocity seismogram.
        """
        self.tr.data = np.gradient(self.tr.data, self.tr.stats.delta)

    def convolve_stf(self, cmt_solution):
        """
        Convolves with a gaussian source time function, with a given
        half_duration. Does this in place. Takes a cmtsolution object as a
        parameter.
        """
        n_convolve = int(math.ceil(2.5 * cmt_solution.half_duration /
                                   self.tr.stats.delta))
        print cmt_solution.half_duration
        print n_convolve
        print 2*n_convolve+1
        # g_x = np.zeros(2 * n_convolve + 1)
        #
        # for i, j in enumerate(range(-n_convolve, n_convolve + 1)):
        #     tau = j * self.tr.stats.delta
        #     exponent = cmt_solution.alpha * cmt_solution.alpha * tau * tau
        #     source = cmt_solution.alpha * math.exp(-exponent) / \
        #         math.sqrt(math.pi)
        #
        #     g_x[i] = source * self.tr.stats.delta

        # self.tr.data = np.convolve(self.tr.data, g_x, 'same')
        # self.data = np.convolve(self.data, g_x, 'same')

    def filter(self, min_period, max_period):
        """
        Performs a bandpass filtering.
        """        
        self.tr.filter('lowpass', freq=(1. / min_period), corners=5,
                       zerophase=True)
        self.tr.filter('highpass', freq=(1. / max_period), corners=2,
                       zerophase=True)
                       
    def write_mseed(self):
        """
        Write an mseed file.
        """
        filename = os.path.join(
            self.directory, self.tr.stats.network + '.' + self.tr.stats.station 
            + '.' + self.tr.stats.location + '.' + self.tr.stats.channel + 
            '.mseed')
        self.tr.write(filename, format="mseed")

    def plot_seismogram(self):
        """
        Plots the seismogram in the time domain.
        """
        self.process_synthetics()
        norm_data = self.normalize()
        time = self.t / 60.
        plt.plot(time, norm_data, 'k')
        plt.xlabel('Time (m)')
        plt.ylabel('Amplitude (normalized)')
        plt.xlim(0, max(time))
        plt.ylim(np.amax(np.absolute(norm_data)) * (-1.2),
                 np.amax(np.absolute(norm_data)) * (1.2))
        plt.title('Plot of ' + self.fname)
        plt.show()
