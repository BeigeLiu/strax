import numpy as np
from .plugin import Plugin, SaveWhen
import strax


export, __all__ = strax.exporter()

@strax.takes_config(
    strax.Option('s1_risetime_area_parameters', type=tuple, default=(50,80,12)),
    strax.Option('s1_risetime_aft_parameters', type=tuple, default=(-1,2.6), track=False),
    strax.Option('s1_flatten_threshold_aft',type=tuple,default=(0.6,100)),
    strax.Option('n_top_pmts', type=int, default=4, track=False),
    strax.Option('s1_max_rise_time_post100',type=int,default=100,track=False),
    strax.Option('s1_min_coincidence',type=int,default=2,track=False),
    strax.Option('s2_min_pmts',type=int,default=2,track=False)
)


@export
class PeakletClassification(Plugin):
    """Classify peaklets as unknown, S1, or S2."""
    __version__ = '3.0.3'

    provides = 'peaklet_classification'
    depends_on = ('peaklets',)
    save_when       = SaveWhen.ALWAYS
    parallel = True
    dtype = (strax.peak_interval_dtype
             + [('type', np.int8, 'Classification of the peak(let)')])

    def setup(self):
        self.s1_risetime_area_parameters = self.config['s1_risetime_area_parameters']

        self.s1_risetime_aft_parameters = self.config['s1_risetime_aft_parameters']

        self.s1_flatten_threshold_aft = self.config['s1_flatten_threshold_aft']

        self.n_top_pmts = self.config['n_top_pmts']

        self.s1_max_rise_time_post100 = self.config['s1_max_rise_time_post100']

        self.s1_min_coincidence = self.config['s1_min_coincidence']

        self.s2_min_pmts = self.config['s2_min_pmts']
        return super().setup()

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
        n_channels = (peaklets['area_per_channel'] > 0).sum(axis=1)
        n_top = self.n_top_pmts
        area_top = peaklets['area_per_channel'][:, :n_top].sum(axis=1)
        area_total = peaklets['area_per_channel'].sum(axis=1)
        area_fraction_top = area_top / area_total

        is_large_s1 = (peaklets['area'] >= 100)
        is_large_s1 &= (rise_time <= self.s1_max_rise_time_post100)
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
