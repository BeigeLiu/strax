from immutabledict import immutabledict
import numba
import numpy as np
from .plugin import Plugin, SaveWhen
import strax
export, __all__ = strax.exporter()
__all__ += ['NO_PULSE_COUNTS']


@export
class PulseProcessing(Plugin):
    """
    Split raw_records into:
     - (tpc) records
     - pulse_counts

    For TPC records, apply basic processing:
        1. Flip, baseline, and integrate the waveform
        2. Find hits, apply linear filter, and zero outside hits.
    
    pulse_counts holds some average information for the individual PMT
    channels for each chunk of raw_records. This includes e.g.
    number of recorded pulses, lone_pulses (pulses which do not
    overlap with any other pulse), or mean values of baseline and
    baseline rms channel.
    """
    __version__ = '0.2.3'

    parallel = 'process'
    rechunk_on_save = immutabledict(
        records=False,
        pulse_counts=True)
    compressor = 'zstd'

    depends_on = 'raw_records'

    provides = ('records','pulse_counts')
    data_kind = {k: k for k in provides}
    save_when = immutabledict(
        records=SaveWhen.TARGET,
        pulse_counts=SaveWhen.ALWAYS,
    )


    baseline_samples = 40

    # PMT pulse processing options
    pmt_pulse_filter = None

    save_outside_hits = (20,20)

    n_tpc_pmts = 8

    check_raw_record_overlaps = True

    allow_sloppy_chunking = False

    hit_min_amplitude = 15

    def infer_dtype(self):
        # Get record_length from the plugin making raw_records
        self.record_length = strax.record_length_from_dtype(
            self.deps['raw_records'].dtype_for('raw_records'))
        dtype = dict()
        for p in self.provides:
            if 'records' in p:
                dtype[p] = strax.record_dtype(self.record_length)
        dtype['pulse_counts'] = pulse_count_dtype(self.n_tpc_pmts)

        return dtype

    def setup(self):
        self.hit_thresholds = self.hit_min_amplitude

    def compute(self, raw_records, start, end):
        # if self.check_raw_record_overlaps:
        #     check_overlaps(raw_records, n_channels=3000)

        # Throw away any non-TPC records; this should only happen for XENON1T
        # converted data
        # print(raw_records['channel'])
        raw_records = raw_records[
            raw_records['channel'] < self.n_tpc_pmts]
        # Convert everything to the records data type -- adds extra fields.
        r = strax.raw_to_records(raw_records)
        del raw_records

        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        strax.zero_out_of_bounds(r)

        baselining_according_to_digitizer(r)
        #strax.baseline(r,
        #               baseline_samples=self.baseline_samples,
        #               allow_sloppy_chunking=self.allow_sloppy_chunking,
        #               flip=True)

        if len(r):
            # Find hits
            # -- before filtering,since this messes with the with the S/N
            hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)
            if self.pmt_pulse_filter:
                # Filter to concentrate the PMT pulses
                strax.filter_records(
                    r, np.array(self.pmt_pulse_filter))

            le, re = self.save_outside_hits
            r = strax.cut_outside_hits(r, hits,
                                       left_extension=le,
                                       right_extension=re)
            # Probably overkill, but just to be sure...
            strax.zero_out_of_bounds(r)
        strax.integrate(r)
        pulse_counts = count_pulses(r, self.n_tpc_pmts)
        pulse_counts['time']    = start
        pulse_counts['endtime'] = end
        #print(r['channel'])
        return dict(records=r,
                    pulse_counts=pulse_counts)

def baselining_according_to_digitizer(raw_records,flip = True):
    for d_i, d in enumerate(raw_records):
        d['data'][:d['length']] = (
            (-1 if flip else 1) * (d['data'][:d['length']] - int(d['baseline'])))

##
# Pulse counting
##
@export
def pulse_count_dtype(n_channels):
    # NB: don't use the dt/length interval dtype, integer types are too small
    # to contain these huge chunk-wide intervals
    return [
        (('Start time of the chunk', 'time'), np.int64),
        (('End time of the chunk', 'endtime'), np.int64),
        (('Number of pulses', 'pulse_count'),
         (np.int64, n_channels)),
        (('Number of lone pulses', 'lone_pulse_count'),
         (np.int64, n_channels)),
        (('Integral of all pulses in ADC_count x samples', 'pulse_area'),
         (np.int64, n_channels)),
        (('Integral of lone pulses in ADC_count x samples', 'lone_pulse_area'),
         (np.int64, n_channels)),
        (('Average baseline', 'baseline_mean'),
         (np.int16, n_channels)),
        (('Average baseline rms', 'baseline_rms_mean'),
         (np.float32, n_channels)),
    ]


def count_pulses(records, n_channels):
    """Return array with one element, with pulse count info from records"""
    if len(records):
        result = np.zeros(1, dtype=pulse_count_dtype(n_channels))
        _count_pulses(records, n_channels, result)
        return result
    return np.zeros(0, dtype=pulse_count_dtype(n_channels))


NO_PULSE_COUNTS = -9999  # Special value required by average_baseline in case counts = 0


@numba.njit(cache=True, nogil=True)
def _count_pulses(records, n_channels, result):
    count = np.zeros(n_channels, dtype=np.int64)
    lone_count = np.zeros(n_channels, dtype=np.int64)
    area = np.zeros(n_channels, dtype=np.int64)
    lone_area = np.zeros(n_channels, dtype=np.int64)

    last_end_seen = 0
    next_start = 0

    # Array of booleans to track whether we are currently in a lone pulse
    # in each channel
    in_lone_pulse = np.zeros(n_channels, dtype=np.bool_)
    baseline_buffer = np.zeros(n_channels, dtype=np.float64)
    baseline_rms_buffer = np.zeros(n_channels, dtype=np.float64)
    for r_i, r in enumerate(records):
        if r_i != len(records) - 1:
            next_start = records[r_i + 1]['time']

        ch = r['channel']
        if ch >= n_channels:
            print('Channel:', ch)
            raise RuntimeError("Out of bounds channel in get_counts!")

        area[ch] += r['area']  # <-- Summing total area in channel
        if r['record_i'] == 0:
            count[ch] += 1
            baseline_buffer[ch] += r['baseline']
            baseline_rms_buffer[ch] += r['baseline_rms']

            if (r['time'] > last_end_seen
                    and r['time'] + r['pulse_length'] * r['dt'] < next_start):
                # This is a lone pulse
                lone_count[ch] += 1
                in_lone_pulse[ch] = True
                lone_area[ch] += r['area']
            else:
                in_lone_pulse[ch] = False

            last_end_seen = max(last_end_seen,
                                r['time'] + r['pulse_length'] * r['dt'])

        elif in_lone_pulse[ch]:
            # This is a subsequent fragment of a lone pulse
            lone_area[ch] += r['area']
    res = result
    res['pulse_count'][:] = count[:]
    res['lone_pulse_count'][:] = lone_count[:]
    res['pulse_area'][:] = area[:]
    res['lone_pulse_area'][:] = lone_area[:]
    means = (baseline_buffer / count)
    means[np.isnan(means)] = NO_PULSE_COUNTS
    res['baseline_mean'][:] = means[:]
    res['baseline_rms_mean'][:] = (baseline_rms_buffer / count)[:]
    return res


##
# Misc
##
@export
@numba.njit(cache=True, nogil=True)
def mask_and_not(x, mask):
    return x[mask], x[~mask]


@export
@numba.njit(cache=True, nogil=True)
def channel_split(rr, first_other_ch):
    """Return """
    return mask_and_not(rr, rr['channel'] < first_other_ch)


@export
def check_overlaps(records, n_channels):
    """Raise a ValueError if any of the pulses in records overlap

    Assumes records is already sorted by time.
    """
    last_end = np.zeros(n_channels, dtype=np.int64)
    channel, time = _check_overlaps(records, last_end)
    if channel != -9999:
        raise ValueError(
            f"Bad data! In channel {channel}, a pulse starts at {time}, "
            f"BEFORE the previous pulse in that same channel ended "
            f"(at {last_end[channel]})")


@numba.njit(cache=True, nogil=True)
def _check_overlaps(records, last_end):
    for r in records:
        # print('==========check overlaps=========')
        # print(r['data'])
        # print('time:',r['time'])
        # print('last end',last_end[r['channel']])
        if r['time'] < last_end[r['channel']]:
            return r['channel'], r['time']
        last_end[r['channel']] = strax.endtime(r)
    return -9999, -9999
