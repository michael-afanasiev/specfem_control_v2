import os
import sys
import time
import math
import obspy
import utils
import obspy
import timeit
import tarfile
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import repeat
from obspy.fdsn import Client
from obspy.signal.rotate import rotate2ZNE
from obspy.signal.invsim import c_sac_taper
from multiprocessing import Pool, cpu_count
from collections import namedtuple

import xml.etree.ElementTree as ET

XML_STRING = '{http://quakeml.org/xmlns/bed/1.2}'
TO_SEC = 3600
NUM_THREADS = 8

def _download_bulk_waveforms(
    (event, networks, stations, start_times, end_times, download_path,
    event_xml_directory, recording_time, padding_time)):
        
    time.sleep(1)
    c = Client("IRIS")
    
    # Make directory.
    utils.mkdir_p(os.path.join(download_path, event))    
    filename = os.path.join(download_path, event, 'data.mseed')
    
    # Don't re-download files.
    if os.path.exists(filename):
        return
            
    # Find start time and end time.
    event_info = os.path.join(event_xml_directory, event + '.xml')
    tree = ET.parse(event_info)
    root = tree.getroot()
    for tag in root.iter():
        if tag.tag == XML_STRING + 'time':
            time.start = obspy.UTCDateTime(
                tag.findall(XML_STRING + 'value')[0].text) - padding_time
            time.end = time.start + recording_time + 2 * padding_time
            break

    # Set up request.
    bulk = []
    for x, y, s_time, e_time in zip(networks, stations, start_times, end_times):
        
        if time.start < s_time or time.start > e_time:
            continue        
        bulk.append((x, y, '*', '*H*', time.start, time.end))
    
    utils.print_ylw('Downloading %s...' % (event))
    c.get_waveforms_bulk(bulk, filename=filename, quality='B')

def station_statistics(params, station_list):
    """
    Looks through the station_list file and makes some statistics on the data. 
    Helpful for deciding downloading parameters.
    """

    # Read the data and parse out the important components.
    stations_list = pd.read_csv(station_list, delimiter='|')
    stations_list.fillna('00', inplace=True)
    networks = stations_list['Network']   
    stations = stations_list['Station']
    sensors = stations_list['SensorDescription']
    locations = stations_list['Location']
    start_times = stations_list['StartTime']
    end_times = stations_list['EndTime']
    
    print str(len(stations)) + " entries."
    start_times = [obspy.UTCDateTime(x).year for x in start_times]
    end_times = [obspy.UTCDateTime(x).year for x in end_times]
    
    plt.hist(
        start_times, bins=range(min(start_times), max(start_times) + 1, 1))
    plt.show()

def download_data(params, station_list, with_waveforms, recording_time, 
                  padding_time):
    """
    Still a work in progress (perhaps never finished). Sorts a text file 
    obtained from IRIS (see manual), and parses out the STS and KS instruments
    (apparently the best ones). Then passes 
    """
    # Domain boundaries
    min_lat = -65
    max_lat = 45
    min_lon = -47.5
    max_lon = 75
        
    # Set up paths and such.
    lasif_data_path = os.path.join(params['lasif_path'], 'DOWNLOADED_DATA')
    event_xml_directory = os.path.join(params['lasif_path'], 'EVENTS')
    event_list = params['event_list']
    lasif_stations_path = os.path.join(params['lasif_path'], 'STATIONS', 
                                       'StationXML')
    
    # Set up station tuple and allowable instruments.
    station = namedtuple('station', ['network', 'station', 'location', 
                         'sensor', 's_time', 'e_time'])
    
    # Read the data and parse out the important components.
    stations_list = pd.read_csv(station_list, delimiter='|')
    stations_list.fillna('00', inplace=True)
    
    # Filter based on domain boundaries
    stations_list = stations_list[stations_list.Latitude > min_lat]
    stations_list = stations_list[stations_list.Latitude < max_lat]
    stations_list = stations_list[stations_list.Longitude > min_lon]
    stations_list = stations_list[stations_list.Longitude < max_lon]
    stations_list = stations_list[stations_list.Location == '00']
    stations_list['StartTime'] = \
        stations_list['StartTime'].astype(obspy.UTCDateTime)
    stations_list['EndTime'] = \
        stations_list['EndTime'].astype(obspy.UTCDateTime)
        
    # Number of events.
    num_events = len(os.listdir(event_xml_directory))
    event_names = sorted([x[:-4] for x in os.listdir(event_xml_directory)])
    
    # Event arrays.
    networks = stations_list.Network
    stations = stations_list.Station
    start_time = stations_list.StartTime
    end_time = stations_list.EndTime

    # Waveforms.
    pool = Pool(processes=NUM_THREADS)
    pool.map(_download_bulk_waveforms, zip(
        event_names, repeat(networks), repeat(stations), repeat(start_time),
        repeat(end_time), repeat(lasif_data_path), repeat(event_xml_directory),
        repeat(recording_time), repeat(padding_time)))    

    if with_waveforms:
        return

    # Get stations.
    c = Client("IRIS")
    for sta, net in zip(stations, networks): 
        station_filename = os.path.join(
            lasif_stations_path, 'station.%s_%s.xml' % (net, sta))            
        if os.path.exists(station_filename):
            continue        
        utils.print_ylw(
            "Downloading StationXML for: %s.%s" % (net, sta))
        try:
            c.get_stations(
                network=net, station=sta, location="*", channel="*",
                level="response", filename=station_filename)
        except:
            utils.print_red("No data for %s" % (station_filename))

def prefilter_data(params):
    """
    Rotates the downloaded data in the "DOWNLOADED" folder. Will fail if no
    StationXML files exist, so make sure you've done this first.
    """
    # Local variables.
    channel_priority = ['BH*', 'LH*']
    lasif_path = params['lasif_path']
    lasif_data_path = os.path.join(params['lasif_path'], 'DOWNLOADED_DATA')
    event_xml_directory = os.path.join(params['lasif_path'], 'EVENTS')
        
    # Get starttime for event.
    for event in sorted(os.listdir(event_xml_directory)):
        
        utils.print_ylw("Prefiltering event: " + event[:-4])                    
        download_path = os.path.join(lasif_data_path, event[:-4])
        
        # Make raw data directory in LASIF project normal spot.
        lasif_raw_data_path = os.path.join(
            lasif_path, 'DATA', event[:-4], 'raw')
        utils.mkdir_p(lasif_raw_data_path)

        st = obspy.read(os.path.join(download_path, 'data.mseed'))
        st_filt = obspy.Stream()
        unique_stations = set([x.stats.station for x in st])
        for s in unique_stations:
            station_components = st.select(station=s)
            for c in channel_priority:
                if len(station_components.select(channel=c)):
                    st_filt += station_components.select(channel=c)
                    break

        write_filename = os.path.join(lasif_raw_data_path, 'data.mseed')
        st_filt.write(filename=write_filename, format='mseed')
        
def convolve_stf(tr):
    """
    Convolves with a gaussian source time function, with a given
    half_duration. Does this in place. Takes a cmtsolution object as a
    parameter.
    """
    half_duration = 10.0
    source_decay_mimic_triangle = 1.6280
    alpha = source_decay_mimic_triangle / half_duration
    n_convolve = int(math.ceil(2.5 * half_duration /
                               tr.stats.delta))
    g_x = np.zeros(2 * n_convolve + 1)

    for i, j in enumerate(range(-n_convolve, n_convolve + 1)):
        tau = j * tr.stats.delta
        exponent = alpha * alpha * tau * tau
        source = alpha * math.exp(-exponent) / \
            math.sqrt(math.pi)

        g_x[i] = source * tr.stats.delta

    return np.convolve(tr.data, g_x, 'same')

def process_synthetics(st, iteration):

    # Filtering frequenceis.
    lowpass_freq = iteration.get_process_params()['lowpass']
    highpass_freq = iteration.get_process_params()['highpass']
    
    print lowpass_freq, highpass_freq
    
    freqmin=highpass_freq
    freqmax=lowpass_freq

    f2 = highpass_freq
    f3 = lowpass_freq
    f1 = 0.8 * f2
    f4 = 1.2 * f3
    pre_filt = (f1, f2, f3, f4)
        
    # Detrend and taper.
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type="hann")

    # Perform a frequency domain taper like during the response removal
    # just without an actual response...
    for tr in st:
        
        tr.data = convolve_stf(tr)
        
        tr.differentiate()
        
        data = tr.data.astype(np.float64)
        orig_len = len(data)

        # smart calculation of nfft dodging large primes
        from obspy.signal.util import _npts2nfft
        nfft = _npts2nfft(len(data))

        fy = 1.0 / (tr.stats.delta * 2.0)
        freqs = np.linspace(0, fy, nfft // 2 + 1)

        # Transform data to Frequency domain
        data = np.fft.rfft(data, n=nfft)
        data *= c_sac_taper(freqs, flimit=pre_filt)
        data[-1] = abs(data[-1]) + 0.0j
        # transform data back into the time domain
        data = np.fft.irfft(data)[0:orig_len]
        # assign processed data and store processing information
        tr.data = data
    
    return st