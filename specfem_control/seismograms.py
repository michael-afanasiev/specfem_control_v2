#!/usr/bin/env python

import os
import math
import utils
import obspy

import numpy as np
import matplotlib.pyplot as plt

class SeismogramNotFoundError(Exception):
    pass


def plot_two(s1, s2):
    
    norm_data_s1 = s1.normalize()
    time_s1 = s1.t / 60.
    
    norm_data_s2 = s2.normalize()
    time_s2 = s2.t / 60.
            
    plt.plot(time_s1, norm_data_s1, 'k', label=s1.fname)
    plt.plot(time_s2, norm_data_s2, 'r', label=s2.fname)
    
    ymax = max(np.amax(np.absolute(norm_data_s1)), 
               np.amax(np.absolute(norm_data_s2))) * (1.2)
    ymin = max(np.amax(np.absolute(norm_data_s1)), 
               np.amax(np.absolute(norm_data_s2))) * (-1.2)
    
    plt.xlabel('Time (m)')
    plt.ylabel('Amplitude (normalized)')
    plt.xlim(0, max(time_s1))
    plt.ylim(ymin, ymax)
    plt.legend()

    plt.show()
    
class Seismogram(object):

    def __init__(self, file_name):

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
            self.tr = obspy.read(file_name)[0]
            self.fname = file_name
            self.t = np.array(
                [x*self.tr.stats.delta for x in range(0, self.tr.stats.npts)])
        else:
            raise SeismogramNotFoundError(utils.print_red("Seismogram not "
                "found."))
                
        

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
        
