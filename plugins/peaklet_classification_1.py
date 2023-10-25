import numpy as np
from .plugin import Plugin, SaveWhen
import strax


export, __all__ = strax.exporter()


@export
class PeakletClassification1(Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    __version__ = '3.0.3'

    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    parallel = True
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')])

    s1_risetime_area_parameters = (50,80,12)

    s1_risetime_aft_parameters = (-1,2.6)

    s1_flatten_threshold_aft = (0.6,100)

    n_top_pmts = 4

    s1_max_rise_time_post100 = 100

    s1_min_coincidence = 2

    width_threshold_s1 = 200 ## in ns

    s2_min_pmts = 2

    @staticmethod
    def upper_rise_time_area_boundary(area, norm, const, tau):
        """
        Function which determines the upper boundary for the rise-time
        for a given area.
        """
        # result = norm * np.exp(-area / tau) + const
        # print(result)
        return norm * np.exp(-area / tau) + const

    @staticmethod
    def upper_rise_time_aft_boundary(aft, slope, offset, aft_boundary, flat_threshold):
        """
        Function which computes the upper rise time boundary as a function
        of area fraction top.
        """
        ## aft: fraction
        res = 10 ** (slope * aft + offset)
        res[aft >= aft_boundary] = flat_threshold
        return res

    def compute(self, peaklets):
        ptype = np.zeros(len(peaklets), dtype=np.int8)

        # Properties needed for classification:
        rise_time = -peaklets['area_decile_from_midpoint'][:, 1]
        width      = peaklets['width']
        n_channels = (peaklets['area_per_channel'] > 0).sum(axis=1)
        n_top = self.n_top_pmts
        area_top = peaklets['area_per_channel'][:, :n_top].sum(axis=1)
        area_total = peaklets['area_per_channel'].sum(axis=1)
        area_fraction_top = area_top / area_total

        is_large_s1 = (peaklets['area'] >= 100)
        is_large_s1 &= (rise_time <= self.s1_max_rise_time_post100)
        is_large_s1 &= (width <= self.width_threshold_s1)
        is_large_s1 &= peaklets['tight_coincidence'] >= self.s1_min_coincidence

        is_small_s1 = peaklets["area"] < 100
        is_small_s1 &= rise_time < self.upper_rise_time_area_boundary(
            peaklets["area"],
            *self.s1_risetime_area_parameters,
        )

        is_small_s1 &= rise_time < self.upper_rise_time_aft_boundary(
            area_fraction_top,
            *self.s1_risetime_aft_parameters,
            *self.s1_flatten_threshold_aft,
        )

        is_small_s1 &= peaklets['tight_coincidence'] >= self.s1_min_coincidence

        ptype[is_large_s1 | is_small_s1] = 1

        is_s2 = n_channels >= self.s2_min_pmts
        is_s2[is_large_s1 | is_small_s1] = False
        ptype[is_s2] = 2

        return dict(type=ptype,
                    time=peaklets['time'],
                    dt=peaklets['dt'],
                    # Channel is added so the field order of the merger of
                    # peaklet_classification and peaklets matches that
                    # of peaklets.
                    # This way S2 merging works on arrays of the same dtype.
                    channel=-1,
                    length=peaklets['length'])
    

def compute_width(peaklets):
    ##### calculate the lower hight width
    return 
