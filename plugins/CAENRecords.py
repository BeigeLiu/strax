import strax
import strax.CAENReader as CAENReader
import numpy as np
from .plugin import Plugin, SaveWhen
import os
from tqdm import tqdm
import sys
##### One Channel Only 
##### TODO： multi channel reader plugin
import time

@strax.takes_config(
    strax.Option('time_range_per_chunk', 
                 type=int, 
                 default=800000000, 
                 track=False,
                 help = 'time range of one chunk to pack (in ns)'),
    strax.Option('raw_record_filename',
                 type=str,
                 default=None,
                 track=False,
                 help = 'filename that provides the raw records')
)


class CAENRecords(Plugin):
    provides        = 'raw_records'
    parallel        = 'process'
    __version__ = '1.0.0'
    depends_on      = ()
    save_when       = SaveWhen.TARGET
    dtype           = strax.raw_record_dtype()
    run_data        = CAENReader.DataFile
    abs_time        = 0
    def setup(self):
        print('current filename: ',self.config['raw_record_filename'])
        self.run_data    = CAENReader.DataFile(fileName=self.config['raw_record_filename'])
        # trigger          = self.run_data.getNextTrigger(chunk_i=0)
        # self.abs_time    = trigger['time']
        # trigger['time'] -= self.abs_time
        #pre,current,next = self.run_data.getNextTrigger()
        return super().setup()

    def source_finished(self):
        return True

    def version(self, run_id=None):
        return self.__version__

    def is_ready(self, chunk_i):
        return int((chunk_i+1)*self.config['time_range_per_chunk']) < self.run_data.end_time

    def check_filename(self):
        if self.config['filename']==None:
            raise ValueError(' Path of DAQ data file is not given')
        if not os.path.exists(self.config['filename']):
            raise FileExistsError(self.config['filename']+' is not found')

    def compute(self,chunk_i):
        triggers = self.run_data.get(self.config['time_range_per_chunk'],chunk_i)
        if len(triggers):
            ##### not empty window
            start   = np.min(triggers['time'])
            end     = np.max(strax.endtime(triggers))
        else:
            ##### empty window
            start   = self.config['time_range_per_chunk']*chunk_i+self.run_data.begin_time
            end     = self.config['time_range_per_chunk']*(chunk_i+1)+self.run_data.begin_time
            triggers= np.zeros(1,dtype=strax.raw_record_dtype(
                samples_per_record=self.run_data.recordLen
            ))
            triggers['time'] = start

        chunk = self.chunk(start=start,end=end,data = triggers)
        # del triggers
        return chunk