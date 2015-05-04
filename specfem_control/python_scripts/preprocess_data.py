#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific function processing observed data.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import obspy
import sys
import os

from lasif import LASIFError
from scipy import signal

# Command line arguments
directory = sys.argv[1]
event = sys.argv[2]
freqmin = float(sys.argv[3])
freqmax = float(sys.argv[4])
origin_time = obspy.UTCDateTime(sys.argv[5])
dt = float(sys.argv[6])
npts = int(sys.argv[7])


def zerophase_chebychev_lowpass_filter(trace, freqmax):
    """
    Custom Chebychev type two zerophase lowpass filter useful for
    decimation filtering.

    This filter is stable up to a reduction in frequency with a factor of
    10. If more reduction is desired, simply decimate in steps.

    Partly based on a filter in ObsPy.

    :param trace: The trace to be filtered.
    :param freqmax: The desired lowpass frequency.

    Will be replaced once ObsPy has a proper decimation filter.
    """
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freqmax / (trace.stats.sampling_rate * 0.5)  # stop band frequency
    wp = ws  # pass band frequency

    while True:
        if order <= 12:
            break
        wp *= 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

    b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

    # Apply twice to get rid of the phase distortion.
    trace.data = signal.filtfilt(b, a, trace.data)

# Read seismograms and gather basic information.
specfem_delta_delay = -1.0687500
starttime = origin_time + specfem_delta_delay
endtime = starttime + dt * npts - 1
duration = endtime - starttime
input_filename = os.path.join(directory, 'DATA', event, 'raw', 'data.mseed')

# Read into obspy stream, and initalize empty processed stream.
st = obspy.read(input_filename)
processed_st = obspy.Stream()

for tr in st:

    # Trim with a short buffer in an attempt to avoid boundary effects.
    # starttime is the origin time of the event
    # endtime is the origin time plus the length of the synthetics
    tr.trim(starttime - 0.2 * duration, endtime + 0.2 * duration)

    # Some basic checks on the data.
    # Non-zero length
    if not len(tr):
        msg = "No data found in time window around the event. File skipped."
        raise LASIFError(msg)

    # No nans or infinity values allowed.
    if not np.isfinite(tr.data).all():
        msg = "Data contains NaNs or Infs. File skipped"
        raise LASIFError(msg)

    # =========================================================================
    # Step 1: Decimation
    # Decimate with the factor closest to the sampling rate of the synthetics.
    # The data is still oversampled by a large amount so there should be no
    # problems. This has to be done here so that the instrument correction is
    # reasonably fast even for input data with a large sampling rate.
    # =========================================================================
    chebyfail = False
    while True:

        decimation_factor = int(dt / tr.stats.delta)
        # Decimate in steps for large sample rate reductions.
        if decimation_factor > 8:
            decimation_factor = 8
        if decimation_factor > 1:
            new_nyquist = tr.stats.sampling_rate / 2.0 / float(
                decimation_factor)
            try:
                zerophase_chebychev_lowpass_filter(tr, new_nyquist)
            except:
                print "CHEBYCHEV FAILED"
                chebyfail = True
            tr.decimate(factor=decimation_factor, no_filter=True)
        else:
            break
    if chebyfail:
        continue

    # =========================================================================
    # Step 2: Detrend and taper.
    # =========================================================================
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.05, max_length=300, type="hann")

    # =========================================================================
    # Step 3: Instrument correction
    # Correct seismograms to velocity in m/s.
    # =========================================================================
    station_name = 'station.%s_%s.xml' % (tr.stats.network, tr.stats.station)
    station_file = os.path.join(
        directory, 'STATIONS', 'StationXML', station_name)

    # check if the station file actually exists
    if not os.path.exists(station_file):
        msg = "No station file found for the relevant time span. File skipped"
        raise LASIFError(msg)

    # This is really necessary as other filters are just not sharp enough
    # and lots of energy from other frequency bands leaks into the frequency
    # band of interest
    f2 = 1.0 * freqmin
    f3 = 1.0 * freqmax
    f1 = 0.8 * f2
    f4 = 1.2 * f3
    pre_filt = (f1, f2, f3, f4)

    try:
        inv = obspy.read_inventory(station_file, format="stationxml")
    except Exception as e:
        msg = ("Could not open StationXML file '%s'. Due to: %s. Will be "
               "skipped." % (station_file, str(e)))
        raise LASIFError(msg)
    tr.attach_response(inv)
    try:
        tr.remove_response(output="VEL", pre_filt=pre_filt,
                           zero_mean=False, taper=False, water_level=1.0)
    except Exception as e:
        msg = ("File  could not be corrected with the help of the "
               "StationXML file '%s'. Due to: '%s'  Will be skipped.") \
            % (station_file, e.__repr__()),
        continue

    # =========================================================================
    # Step 5: Interpolation
    # =========================================================================
    # Make sure that the data array is at least as long as the
    # synthetics array. Also add some buffer sample for the
    # spline interpolation to work in any case.
    buf = dt * 5
    if starttime < (tr.stats.starttime + buf):
        tr.trim(starttime=starttime - buf, pad=True, fill_value=0.0)
    if endtime > (tr.stats.endtime - buf):
        tr.trim(endtime=endtime + buf, pad=True, fill_value=0.0)

    try:
        tr.interpolate(
            sampling_rate=1.0 / dt,
            method="weighted_average_slopes", starttime=starttime,
            npts=npts)
    except:
        print "INTERPOLATION FAILED."
        continue

    # =========================================================================
    # Save processed data and clean up.
    # =========================================================================
    # Convert to single precision to save some space.
    tr.data = np.require(tr.data, dtype="float32", requirements="C")
    if hasattr(tr.stats, "mseed"):
        tr.stats.mseed.encoding = "FLOAT32"

    # Add trace to processed stream
    processed_st += tr

# Save processed stream. Make directory if it does not exist.
processed_dir = os.path.join(
    directory, 'DATA', event, 'preprocessed_%s_%s' %
    (1 / freqmax, 1 / freqmin))
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)
processed_st.write(
    os.path.join(processed_dir, 'data.mseed'), format='mseed')
