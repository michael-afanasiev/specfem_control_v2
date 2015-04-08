import os
import sys
import time
import obspy
import utils
import obspy
import timeit
import tarfile
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

from itertools import repeat
from obspy.fdsn import Client
from obspy.signal.rotate import rotate2ZNE
from multiprocessing import Pool, cpu_count
from collections import namedtuple

import xml.etree.ElementTree as ET

XML_STRING = '{http://quakeml.org/xmlns/bed/1.2}'
TO_SEC = 3600
NUM_THREADS = 8

def _download_files_new(bulk, download_path):
        
    time.sleep(1)
    c = Client("IRIS")
    
    filename = download_path + "/data.mseed"
    
    # Attempt to download all valid locations. TODO clean up the location array.
    c.get_waveforms_bulk(bulk, filename=filename)
            
            
def _download_files((network, station, location, starttime, endtime, 
                    download_path)):
    
    time.sleep(1)
    c = Client("IRIS")
    
    filename = download_path + "/%s.%s.%s.mseed" % (network, station, location)

    # Attempt to download all valid locations. TODO clean up the location array.
    found = False
    for location in ['', '00', '10', '20', '30', '01', '02', '03']:
        try:
            c.get_waveforms(
                network=network, station=station,
                location=location, channel='?H?',
                starttime=starttime, endtime=endtime,
                filename=filename)
            found = True
            break
        except:
            continue
            
    if found == False:
        print "Data non-existatant for: %s.%s" % (network, station)
        return
                 
    # Split full .mseed file into components.                                 
    tr = obspy.read(filename)
    for trace in tr:
        try:
            trace.write(
                download_path + "/%s.%s.%s.%s.mseed" % 
                (trace.stats.network, trace.stats.station, trace.stats.location, 
                trace.stats.channel), format='mseed')
        except:
            print "Bad component for station: %s.%s.%s.%s" % \
                (trace.stats.network, trace.stats.station, trace.stats.location, 
                trace.stats.channel)
            continue
            
    os.remove(filename)
    
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
    
    print str(len(stations)) + " entries."
    start_times = [obspy.UTCDateTime(x).year for x in start_times]
    
    plt.hist(
        start_times, bins=range(min(start_times), max(start_times) + 1,
        1))
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
    
    # Desired instruments.
    instruments = ['STS', 'KS']
        
    lasif_data_path = os.path.join(params['lasif_path'], 'DOWNLOADED_DATA')
    iteration_xml_path = params['iteration_xml_path']
    event_xml_directory = os.path.join(params['lasif_path'], 'EVENTS')
    event_list = params['event_list']
    lasif_stations_path = os.path.join(params['lasif_path'], 'STATIONS', 
                                       'StationXML')
    
    # Set up station tuple and allowable instruments.
    station = namedtuple('station', ['network', 'station', 'location', 
                         'sensor', 's_time', 'e_time'])
    time = namedtuple('time', ['start', 'end'])
    
    # Read the data and parse out the important components.
    stations_list = pd.read_csv(station_list, delimiter='|')
    
    # Filter based on domain boundaries
    stations_list = stations_list[stations_list.Latitude > min_lat]
    stations_list = stations_list[stations_list.Latitude < max_lat]
    stations_list = stations_list[stations_list.Longitude > min_lon]
    stations_list = stations_list[stations_list.Longitude < max_lon]
        
    stations_list.fillna('00', inplace=True)
    networks = stations_list['Network']   
    stations = stations_list['Station']
    sensors = stations_list['SensorDescription']
    locations = stations_list['Location']    
    
    # Filtered start times.
    start_times = stations_list['StartTime']
    start_times = [obspy.UTCDateTime(x) for x in start_times]
    end_times = stations_list['EndTime']
    end_times = [obspy.UTCDateTime(x) for x in end_times]    
    
    # Filter the stations based on instrumentation.
    stations_filt = []
    for net, sta, sen, loc, s_time, e_time in zip(
        networks, stations, sensors, locations, start_times, end_times):
        if (loc == '00'):
            if not stations_filt:
                stations_filt.append(
                    station(network=net, station=sta, location=loc, 
                            sensor=sen, s_time=s_time, e_time=e_time))
            elif stations_filt[-1].station != sta:
                stations_filt.append(
                    station(network=net, station=sta, location=loc, 
                            sensor=sen, s_time=s_time, e_time=e_time))                                                     
    utils.print_blu("Found %i stations." % (len(stations_filt)))

    # Make download event data directories.
    utils.mkdir_p(lasif_data_path)
    for event in event_list:
        utils.mkdir_p(os.path.join(lasif_data_path, event))
    
    # Number of events.
    num_events = len(os.listdir(event_xml_directory))
    
    # For each event...
    for i, event in enumerate(sorted(os.listdir(event_xml_directory))):
        
        # Make download event data directories if they weren't.
        utils.mkdir_p(os.path.join(lasif_data_path, event[:-4]))
        
        # Start the timer and set path.
        start_time = timeit.default_timer()
        download_path = os.path.join(lasif_data_path, event[:-4])
                
        # Skip those for which files are already downloaded.
        if 'data.tar' in os.listdir(download_path):
            continue
        utils.print_ylw('[' + str(i+1) + '/' + str(num_events) + ']\t\t'
            'Downloading data for: ' + event[:-4])
        
        # Find start time and end time.
        event_info = os.path.join(event_xml_directory, event)
        tree = ET.parse(event_info)
        root = tree.getroot()
        for tag in root.iter():
            if tag.tag == XML_STRING + 'time':
                time.start = obspy.UTCDateTime(
                    tag.findall(XML_STRING + 'value')[0].text) - padding_time
                time.end = time.start + recording_time + 2 * padding_time
                break

        # Download if requested (optimized for parallel).
        if with_waveforms:
                        
            # Make sure stations exist for the right time.
            stations_filt_this = [x for x in stations_filt if 
                                  (time.start > x.s_time) 
                                  and (time.start < x.e_time)]                            
                                  
            print "%d stations in the right time interval." % \
                (len(stations_filt_this))

            if __name__ == "specfem_control.data":
                
                pool = Pool(processes=NUM_THREADS)
                net_pass = [x.network for x in stations_filt_this]
                sta_pass = [x.station for x in stations_filt_this]
                loc_pass = [x.location for x in stations_filt_this]
                
                bulk = []
                for x, y in zip(net_pass, sta_pass):
                    bulk.append((x, y, "*", "*H*", time.start, time.end))
                
                print
                _download_files_new(bulk, download_path)

                sys.exit()
                # pool.map(
                #     _download_files, zip(net_pass, sta_pass, loc_pass,
                #     repeat(time.start), repeat(time.end),
                #     repeat(download_path)))

            # Tar and delete.
            tar = tarfile.open(os.path.join(download_path, "data.tar"), "w")
            for name in os.listdir(download_path):
                tar.add(os.path.join(download_path, name), arcname=name)
            tar.close()
            for name in os.listdir(download_path):
                if not name.endswith('.tar'):
                    os.remove(os.path.join(download_path, name))
        
            # Stop timer.            
            end_time = timeit.default_timer()
            utils.print_cyn("This event took %d seconds to download. At this "
                            "pace you'll be done in about %f hours." % 
                            (end_time - start_time, 
                            (num_events - i) * (end_time - start_time) 
                            / float(TO_SEC)))
                        
    utils.print_blu('Done (or skipping) downloading.')
    if with_waveforms:
        return

    c = Client("IRIS")
    for x in stations_filt:
        
        station_filename = os.path.join(
            lasif_stations_path, 'station.%s_%s.xml' % (x.network, x.station))
            
        if os.path.exists(station_filename):
            continue
        
        utils.print_ylw(
            "Downloading StationXML for: %s.%s" % (x.network, x.station))
        try:
            c.get_stations(
                network=x.network, station=x.station, location="*", channel="*",
                level="response", filename=station_filename)
        except:
            utils.print_red("No data for %s" % (station_filename))
            
def prefilter_data(params):
    """
    Rotates the downloaded data in the "DOWNLOADED" folder. Will fail if no
    StationXML files exist, so make sure you've done this first.
    """
    # Local variables.
    proper_components = ['BHZ', 'BHE', 'BHN']
    azi_dict = {'90.0' : 'BHE', '0.0' : 'BHN'}
    lasif_path = params['lasif_path']
    lasif_data_path = os.path.join(params['lasif_path'], 'DOWNLOADED_DATA')
    event_xml_directory = os.path.join(params['lasif_path'], 'EVENTS')
    lasif_stations_path = os.path.join(
        params['lasif_path'], 'STATIONS', 'StationXML')
        
    # Get starttime for event.
    for event in sorted(os.listdir(event_xml_directory)):
        
        fixed = 0
        utils.print_ylw("Prefiltering event: " + event[:-4])
        
        event_info = os.path.join(event_xml_directory, event)
        tree = ET.parse(event_info)
        root = tree.getroot()
        for tag in root.iter():
            if tag.tag == XML_STRING + 'time':
                starttime = obspy.UTCDateTime(
                    tag.findall(XML_STRING + 'value')[0].text)
                break
                    
        download_path = os.path.join(lasif_data_path, event[:-4])
        
        # Make raw data directory in LASIF project normal spot.
        lasif_raw_data_path = os.path.join(
            lasif_path, 'DATA', event[:-4], 'raw')
        utils.mkdir_p(lasif_raw_data_path)
        if os.path.exists(os.path.join(lasif_raw_data_path, 'data.tar')):
            print "Data already exists for " + event[:-4] + ". Skipping."
            continue
        
        # Untar.
        print "Untarring..."
        tar_path = os.path.join(download_path, 'data.tar')
        tar = tarfile.open(tar_path)
        tar.extractall(path=download_path)
        tar.close()
        os.remove(tar_path)

        # Get inventory from StationXML.
        print "Fixing components..."
        for file_xml in os.listdir(lasif_stations_path):

            # Set up arrays.
            new_components = []
            full_station = []
            traces = []

            # Find names from station file.
            net_sta = file_xml.split('.')
            net_xml, sta_xml = net_sta[1].split('_')

            # Find matching data.
            for file in sorted(os.listdir(download_path)):
                net, sta = file.split('.')[0:2]
                if (net == net_xml) and (sta == sta_xml):
                    full_station.append(os.path.join(download_path, file))

            # Get the data from the three existing components.
            traces = [obspy.read(file)[0] for file in full_station]

            # Open StationXML file and get inventory.
            stationXML = os.path.join(lasif_stations_path, file_xml)
            inv = obspy.read_inventory(stationXML, format='stationxml')

            # Select specific inventory components. Kill those which are not
            # properly aligned, and rename those which are.
            for t in traces:
                if t.stats.channel in proper_components:
                    new_components.append(t.stats.channel)
                else:
                    try:
                        my_inv = inv.select(
                            channel=t.stats.channel, station=t.stats.station,
                            location=t.stats.location, time=t.stats.starttime)
                        new_components.append(
                            azi_dict[str(my_inv[0][0][0].azimuth)])
                        fixed += 1
                    except:
                        new_components.append('delete')

            # Write the newly named files to the raw data lasif folder. The
            # files with a proper N/S orientation should now be called by their
            # proper names.
            for chan, trace in zip(new_components, traces):
                if chan == 'delete':
                    continue
                write_name = os.path.join(
                    lasif_raw_data_path, "%s.%s.%s.%s.mseed" %
                    (trace.stats.network, trace.stats.station,
                    trace.stats.location, chan))
                trace.write(write_name, format='mseed')

        utils.print_cyn("Fixed %d channels." % (fixed))
        
        # Retar.
        print "Tarring downloaded data..."
        with tarfile.open(tar_path, "w") as tar:
            for file in os.listdir(download_path):
                tar.add(os.path.join(download_path, file), arcname=file)
        for file in os.listdir(download_path):
            if not file.endswith('.tar'):
                os.remove(os.path.join(download_path, file))
                
        print "Tarring prefiltered data..."
        rawdata_tar = os.path.join(lasif_raw_data_path, 'data.tar')
        with tarfile.open(rawdata_tar, "w") as tar:
            for file in os.listdir(lasif_raw_data_path):
                tar.add(os.path.join(lasif_raw_data_path, file), arcname=file)
        for file in os.listdir(lasif_raw_data_path):
            if not file.endswith('.tar'):
                os.remove(os.path.join(lasif_raw_data_path, file))        
